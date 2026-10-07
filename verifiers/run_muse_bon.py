"""run_muse_bon.py — Muse-as-BASE Best-of-N generator for the base/verifier swap 2x2.

DRAFT. Produces a drop-in `predictions.jsonl` where each record carries N Muse
Spark 1.1 trajectories (sampled at temperature>0), so the existing post-hoc
juries (verify_jury_local.py / verify_jury_holistic.py / verify_jury_gpt.py) can
select among them exactly as they do for the Qwen-27B BoN runs. This is the
"strong base" arm that was missing from the {weak,strong}base x {weak,strong}verifier
matrix.

It is built faithfully on run_muse.py + methods/muse_backend.py + methods/best_of_n.py:
  - same MuseVLM backend (Responses API, proxy stripped, 32768-token reasoning floor);
  - same canonical benchmarks.load_benchmark (first-N == the shared subset);
  - same prompts.build_qa_prompt / prompts.extract_answer_letter;
  - same record schema as run_muse.py's `work()` PLUS best_of_n._finalize's
    `trajectories` list + majority-vote top-level `prediction`.

Difference vs run_muse.py: per question we call MuseVLM.generate N times at
temperature=0.8 (Muse has NO seed knob — `seed` is stored but never sent — so
diversity comes purely from stochastic sampling; temperature MUST be > 0). Frames
are decoded ONCE per question and reused across all N calls (identical visual
context; only sampling differs).

Requires MODEL_API_KEY (or MUSE_API_KEY) — the Meta Model API LaMa token
(format LLM|<appid>|<secret>). No GPU: Muse runs remotely.

Usage:
  # smoke (first 2 questions x N trajectories)
  MODEL_API_KEY=... python verifiers/run_muse_bon.py \
      --exp-dir <REGISTERED_EXP_ROOT> --bench video-mme-v2 --limit 2

  # full first-300 VMME-v2, N=8 @ T=0.8
  MODEL_API_KEY=... python verifiers/run_muse_bon.py \
      --exp-dir <REGISTERED_EXP_ROOT> --bench video-mme-v2 --n-samples 300 \
      --n-traj 8 --temperature 0.8 --workers 16

Writes to <exp-dir>/eval/<bench>/step_000000/predictions.jsonl (resumable: a
(video_id, question_id) is skipped only when it already has >= N trajectories with
at least one non-error trajectory; errored/partial questions are retried).
"""
from __future__ import annotations

import os

# The internal HTTPS proxy at localhost:10054 mangles HTTPS-CONNECT; api.meta.ai
# is reachable directly. Strip proxy vars BEFORE any networking-capable import,
# mirroring run_muse.py / muse_backend.py / video_io.py.
for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_v, None)

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Reuse the exact primitives the Muse study already ships (same import pattern as
# verify_jury_holistic.py). importing methods.vlm_backend pulls torch (needed by
# VideoReader anyway) but NOT vllm (that is lazy inside VLM.__init__), so this
# runs on the login node with no GPU.
CODE = os.environ["VRITS_ROOT"] + "/experiments/109_muse_zeroshot_vmmev2_500/code"
DEFAULT_CACHE = os.environ["VRITS_DATA"] + "/video_cache"
sys.path.insert(0, CODE)

from benchmarks import load_benchmark                # noqa: E402
from methods.muse_backend import MuseVLM             # noqa: E402
from methods.vlm_backend import frames_to_pil        # noqa: E402
from prompts import build_qa_prompt, extract_answer_letter  # noqa: E402
from video_io import VideoReader                     # noqa: E402


def _is_done(rec: Dict[str, Any], n_traj: int) -> bool:
    """A question is done only once it has >= N trajectories with at least one
    non-error trajectory. Partial writes (process killed mid-question) or
    all-error questions are retried on the next launch — same spirit as
    run_muse.py's 'retry empty/[api error]' resume rule, at trajectory grain."""
    trs = rec.get("trajectories") or []
    if len(trs) < n_traj:
        return False
    n_ok = 0
    for t in trs:
        r = str(t.get("reasoning", ""))
        if r and not r.lower().startswith("[api error") and not r.startswith("ERROR:"):
            n_ok += 1
    return n_ok > 0


def _key(video_id, question_id, question):
    """Unique per-question key. question_id may be None (e.g. TempCompass:
    many questions per video, or no per-question id) -> fall back to question text so
    distinct questions on the same video never collide (mirrors the jury scripts' _key)."""
    return (video_id, question_id if question_id not in (None, "") else question)


def _assemble(s: Dict[str, Any], trajs: List[Dict[str, Any]], secs: float) -> Dict[str, Any]:
    """Build one drop-in predictions.jsonl record.

    Top-level shape mirrors run_muse.py `work()`; `trajectories` + majority-vote
    `prediction` mirror best_of_n._finalize. Keys the juries read:
      video_id, question_id (top-level == meta.question_id), answer_letter,
      meta.{question, options}, trajectories[].{prediction, reasoning}.
    """
    letters = [t["prediction"].strip().upper() for t in trajs if str(t["prediction"]).strip()]
    pred = Counter(letters).most_common(1)[0][0] if letters else ""
    return {
        "video_id": s["video_id"],
        # Top-level question_id = the unique key (question-text fallback for benchmarks
        # with no per-question id, e.g. TempCompass) so juries + resume never collide.
        "question_id": _key(s["video_id"], s.get("meta", {}).get("question_id"), s["question"])[1],
        "prediction": pred,                                  # default = majority vote
        "answer_letter": s["answer_letter"],
        "task_type": s["task_type"],
        "reasoning": trajs[0]["reasoning"] if trajs else "",
        "n_tokens": sum(int(t["n_tokens"]) for t in trajs),
        "n_frames_used": trajs[0]["n_frames_used"] if trajs else 0,
        "time_seconds": secs,
        "tool_calls": [],
        "trajectories": trajs,                               # ALL N — for the post-hoc juries
        "meta": {**s.get("meta", {}), "options": s["options"], "question": s["question"]},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True,
                    help="Registered experiment ROOT (created via profai-cli create-experiment). "
                         "predictions.jsonl is written under <exp-dir>/eval/<bench>/step_000000/.")
    ap.add_argument("--bench", default="video-mme-v2")
    ap.add_argument("--n-samples", type=int, default=300,
                    help="First-N questions of the benchmark = the shared subset for all 2x2 cells.")
    ap.add_argument("--subset", default="")
    ap.add_argument("--with-subtitles", action="store_true")
    ap.add_argument("--n-traj", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="MUST be > 0 for BoN diversity (Muse has no seed control).")
    ap.add_argument("--top-p", type=float, default=1.0)  # accepted by MuseVLM but ignored by the API
    ap.add_argument("--max-new-tokens", type=int, default=4096,
                    help="Floored to 32768 internally by MuseVLM (reasoning headroom).")
    ap.add_argument("--n-frames", type=int, default=32)
    ap.add_argument("--frame-resize", type=int, default=336)
    ap.add_argument("--frame-extractor", default="decord")
    ap.add_argument("--cache-root", default=DEFAULT_CACHE)
    ap.add_argument("--model", default="muse-spark-1.1-eval")
    ap.add_argument("--no-thinking", action="store_true",
                    help="Disable the Muse 'think step by step' preamble (default: enabled, matches "
                         "the exp-109 Muse zeroshot baseline).")
    ap.add_argument("--workers", type=int, default=16,
                    help="Concurrent Responses-API calls. Watch API rate limits: N-way fan-out "
                         "multiplies request rate.")
    ap.add_argument("--inflight", type=int, default=24,
                    help="Questions decoded/queued at a time (bounds RAM: inflight * n_frames images).")
    ap.add_argument("--limit", type=int, default=10 ** 9,
                    help="Cap questions processed this run (for smoke). Also caps benchmark load size.")
    args = ap.parse_args()

    if not (os.environ.get("MODEL_API_KEY") or os.environ.get("MUSE_API_KEY")):
        sys.exit("MODEL_API_KEY not set — export the Meta Model API LaMa token "
                 "(format LLM|<appid>|<secret>) before running.")

    out_dir = Path(args.exp_dir).resolve() / "eval" / args.bench / "step_000000"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "predictions.jsonl"

    # ---- resume ----------------------------------------------------------
    done: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    if preds_path.exists():
        for line in open(preds_path):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Last write wins (juries build a dict too), so duplicate keys are fine.
            if _is_done(r, args.n_traj):
                done[(r.get("video_id"), r.get("question_id"))] = r
    print(f"resuming: {len(done)} already done", flush=True)

    # ---- benchmark (first-N == shared subset) ----------------------------
    # Cap the load so a --limit smoke does not unzip all n_samples videos.
    n_load = min(args.n_samples, args.limit)
    kw: Dict[str, Any] = {"n_samples": n_load}
    if args.subset:
        kw["subset"] = args.subset
    if args.with_subtitles:
        kw["include_subtitle"] = True
    samples = list(load_benchmark(args.bench, cache_root=args.cache_root, **kw))
    todo = [s for s in samples
            if _key(s["video_id"], s.get("meta", {}).get("question_id"), s["question"]) not in done][:args.limit]
    print(f"to process: {len(todo)} questions x {args.n_traj} trajectories "
          f"@ T={args.temperature}", flush=True)
    if not todo:
        print("nothing to do.", flush=True)
        return

    # ---- backend ---------------------------------------------------------
    vlm = MuseVLM(model=args.model, thinking_mode=not args.no_thinking)  # stateless -> thread-safe
    reader = VideoReader(cache_root=args.cache_root,
                         frame_extractor=args.frame_extractor,
                         frame_resize=args.frame_resize)

    def _one(images, prompt, sample_idx: int) -> Tuple[int, Dict[str, Any]]:
        try:
            text, n_tok = vlm.generate(images=images, text=prompt,
                                       max_new_tokens=args.max_new_tokens,
                                       temperature=args.temperature, top_p=args.top_p)
        except Exception as e:  # keep the trajectory slot; empty prediction, error reasoning
            text, n_tok = f"ERROR: {e!r}", 0
        return sample_idx, {
            "prediction": extract_answer_letter(text) or "",
            "reasoning": text,
            "n_tokens": n_tok,
            "n_frames_used": len(images),
            "tool_calls": [],
        }

    records = list(done.values())
    written = 0
    t_start = time.time()
    fout = open(preds_path, "a")
    ex = ThreadPoolExecutor(max_workers=args.workers)

    # Process in question-windows so at most `inflight` questions' frames are
    # resident; the pool stays saturated (inflight * n_traj queued futures).
    for c0 in range(0, len(todo), args.inflight):
        window = todo[c0:c0 + args.inflight]
        qmeta: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
        fut2q = {}
        for s in window:
            qkey = _key(s["video_id"], s.get("meta", {}).get("question_id"), s["question"])
            try:
                frames, _ = reader.load_and_sample(s["video_path"], args.n_frames)
                images = frames_to_pil(frames)  # decode ONCE, reuse across N calls
            except Exception as e:
                # Do not write a placeholder — leave the question un-done so it is
                # retried next launch (a persistently missing video just logs each run).
                print(f"  decode-fail {qkey}: {e!r}", flush=True)
                continue
            prompt = build_qa_prompt(benchmark=args.bench, question=s["question"],
                                     options=s["options"], subtitle=s.get("subtitle"),
                                     include_cot=True)
            qmeta[qkey] = {"s": s, "t0": time.time(), "results": {}}
            for k in range(args.n_traj):
                f = ex.submit(_one, images, prompt, k)
                fut2q[f] = qkey

        for f in as_completed(list(fut2q.keys())):
            qkey = fut2q[f]
            sidx, traj = f.result()
            qm = qmeta[qkey]
            qm["results"][sidx] = traj
            if len(qm["results"]) == args.n_traj:
                trajs = [qm["results"][i] for i in range(args.n_traj)]
                rec = _assemble(qm["s"], trajs, time.time() - qm["t0"])
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                records.append(rec)
                written += 1

        acc = sum(1 for r in records
                  if (r["prediction"] or "").strip().upper() ==
                     (r["answer_letter"] or "").strip().upper()) / max(len(records), 1)
        print(f"  windows done up to {min(c0 + args.inflight, len(todo))}/{len(todo)}  "
              f"written={written}  majority_acc={acc:.4f}  "
              f"elapsed={(time.time() - t_start) / 60:.1f}m", flush=True)

    ex.shutdown(wait=True)
    fout.close()
    print(f"DONE wrote {written} new records to {preds_path}", flush=True)


if __name__ == "__main__":
    main()
