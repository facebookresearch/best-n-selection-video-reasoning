"""Executable evidence verifier (Mechanism B) — post-hoc over a stored best-of-N run.

For each question with >=2 distinct candidate answers, a BLIND check-writer VLM
(Muse Spark 1.1; sees question + options + candidate reasoning + the primitive API
doc — NOT the frames) writes a Python "unit test" per distinct candidate that
assigns a float `score` by calling CLIP-based perception primitives against the real
video (clip_primitives.py / sandbox.py). The candidate with the highest executed
score wins, provided the top margin clears a noise band; otherwise — or on a hard
capability-gate for counting/spatial/audio questions — the verifier abstains to the
majority vote. This is the "sound oracle" contrast to the free-form frame-jury
(verify_jury_mm.py): the CODE reads the pixels, not a fallible VLM judge.

Phase 1 (parallel, API-bound): write + persist snippets -> clip_check_writes.jsonl.
Phase 2 (serial, main thread for signal timeout): execute + pick -> clip_check_picks.jsonl.
Both phases are resumable.

Usage:
  MODEL_API_KEY=... python verify_clip_checks.py <exp_dir> [--bench video-mme-v2]
     [--n-samples 500] [--subset ""] [--workers 4] [--limit N] [--n-pool 16] [--device cpu]
"""
from __future__ import annotations

import os
import re
import sys
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_v, None)

# Shared infra (MuseVLM, VideoReader, load_benchmark, frames_to_pil, uniform_sample)
CODE = os.environ["VRITS_ROOT"] + "/experiments/109_muse_zeroshot_vmmev2_500/code"
CACHE = os.environ["VRITS_DATA"] + "/video_cache"
sys.path.insert(0, CODE)                    # infra dir
sys.path.insert(0, os.path.dirname(__file__))  # this dir (clip_primitives, sandbox)

from methods.muse_backend import MuseVLM     # noqa: E402
from video_io import VideoReader             # noqa: E402
from benchmarks import load_benchmark        # noqa: E402

from clip_primitives import VideoPrimitives  # noqa: E402
from sandbox import run_check                 # noqa: E402

# Hard capability-gate: CLIP cannot judge these -> abstain to majority (never regress).
_GATE = ("count", "spatial", "audio")
_MARGIN = 0.02   # top1-top2 must exceed this to trust the executable pick
_BLOCK = re.compile(r"#{2,3}\s*CHECK\s*\(?([A-H])\)?.*?```(?:python)?\s*(.*?)```",
                    re.DOTALL | re.IGNORECASE)

_API_DOC = """You may call ONLY these functions (plus max, min, abs, round, sum, len, and arithmetic):
  clip(text: str) -> float
      How strongly the video's best-matching frame matches the description `text`
      (CLIP cosine; typical ~0.15-0.35, higher = stronger visual match).
  clip_at(text: str, t0: float, t1: float) -> float
      Same, restricted to the segment from t0 to t1 seconds.
  duration() -> float
      Total video length in seconds.
clip() is reliable for the PRESENCE of a concrete visible object/person/color/scene;
it is NOT reliable for counting, left/right position, reading text, audio, or temporal order."""


def _argval(flag, default):
    return type(default)(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def _is_gated(task_type: str) -> bool:
    t = (task_type or "").lower()
    return any(g in t for g in _GATE)


def _distinct_letters(rec):
    L = [str(t.get("prediction", "")).strip().upper() for t in rec.get("trajectories", [])
         if str(t.get("prediction", "")).strip()]
    return L


def _writer_prompt(rec) -> str:
    m = rec.get("meta", {})
    q = m.get("question", "")
    opts = m.get("options", []) or []
    L = _distinct_letters(rec)
    cnt = Counter(L)
    distinct = sorted(cnt)
    # one reasoning excerpt per distinct candidate (first trajectory concluding it)
    excerpt = {}
    for t in rec.get("trajectories", []):
        p = str(t.get("prediction", "")).strip().upper()
        if p in distinct and p not in excerpt:
            excerpt[p] = str(t.get("reasoning", "")).strip().replace("\n", " ")[:400]
    cands = "\n".join(f"  {d} (chosen by {cnt[d]}/{len(L)} runs): {excerpt.get(d,'')}" for d in distinct)
    return (
        "You are writing executable perception checks to decide which candidate answer to a "
        "long-video multiple-choice question is correct. For EACH distinct candidate letter, write "
        "a short Python snippet that assigns a float `score` = (CLIP match of the key VISIBLE evidence "
        "that this option implies) minus (the best CLIP match of the competing options' evidence). "
        "A higher score means stronger visual support.\n\n"
        "Rules:\n"
        "- Use ONLY the functions in the API below.\n"
        "- Phrase claims as short, concrete descriptions of what would be SEEN if the option were true.\n"
        "- If the option's correctness cannot be judged from a visible object/scene (it hinges on "
        "counting, spatial position, audio, reading text, or temporal order), set `score = 0.0`.\n"
        "- Output EXACTLY one block per distinct candidate letter, in this format:\n"
        "### CHECK <LETTER>\n```python\n<code that assigns score>\n```\n\n"
        f"{_API_DOC}\n\n"
        "Worked example (question: \"What colour is the protagonist's car?\"):\n"
        "### CHECK A\n```python\n# A claims the car is red.\n"
        "score = clip(\"a red car\") - max(clip(\"a blue car\"), clip(\"a white car\"), clip(\"a black car\"))\n```\n\n"
        "Now the real task.\n"
        f"Question: {q}\nOptions:\n" + "\n".join(opts) +
        f"\n\nCandidate analyses (letter = the option each independent run concluded):\n{cands}\n\n"
        f"Write one ### CHECK block for each of these letters: {', '.join(distinct)}"
    )


def _parse_snippets(raw: str) -> dict:
    out = {}
    for letter, code in _BLOCK.findall(raw or ""):
        out[letter.upper()] = code.strip()
    return out


def main():
    exp = sys.argv[1].rstrip("/")
    bench = _argval("--bench", "video-mme-v2")
    n_samples = _argval("--n-samples", 500)
    subset = _argval("--subset", "")
    workers = _argval("--workers", 4)
    limit = _argval("--limit", 10 ** 9)
    n_pool = _argval("--n-pool", 16)
    device = _argval("--device", "")

    pred = sorted(Path(exp).glob("eval/**/predictions.jsonl"))[0]
    writes_p = pred.parent / "clip_check_writes.jsonl"
    picks_p = pred.parent / "clip_check_picks.jsonl"

    # dedup source records by (vid,qid), keep last with trajectories
    recs = {}
    for line in open(pred):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("trajectories"):
            recs[(r.get("video_id"), r.get("question_id"))] = r

    # (vid,qid) -> video_path
    smap = {}
    _bkw = {"subset": subset} if subset else {}
    for s in load_benchmark(bench, cache_root=CACHE, n_samples=n_samples, **_bkw):
        smap[(s["video_id"], s.get("meta", {}).get("question_id"))] = s

    # ---------------------------------------------------------------- Phase 1: write
    writes = {}
    if writes_p.exists():
        for line in open(writes_p):
            try:
                w = json.loads(line); writes[(w["video_id"], w["question_id"])] = w
            except Exception:
                pass
    need = [(k, r) for k, r in recs.items()
            if k in smap and not _is_gated(r.get("task_type", ""))
            and len(set(_distinct_letters(r))) >= 2 and k not in writes][:limit]
    print(f"WRITE phase: {len(recs)} recs, {len(smap)} bench, {len(writes)} written, {len(need)} to write", flush=True)

    if need:
        vlm = MuseVLM(model="muse-spark-1.1-eval", thinking_mode=False, seed=42)

        def _write(k, r):
            raw, ntok = vlm.generate(images=[], text=_writer_prompt(r),
                                     max_new_tokens=8192, temperature=0.0)
            return {"video_id": k[0], "question_id": k[1],
                    "snippets": _parse_snippets(raw), "n_tokens": ntok,
                    "raw_len": len(raw or "")}

        with open(writes_p, "a") as f, ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_write, k, r) for k, r in need]
            for i, fut in enumerate(as_completed(futs)):
                w = fut.result()
                f.write(json.dumps(w) + "\n"); f.flush()
                writes[(w["video_id"], w["question_id"])] = w
                if (i + 1) % 25 == 0:
                    print(f"  wrote {i+1}/{len(need)}", flush=True)

    # ---------------------------------------------------------------- Phase 2: execute (serial)
    picks = {}
    if picks_p.exists():
        for line in open(picks_p):
            try:
                p = json.loads(line); picks[(p["video_id"], p["question_id"])] = p
            except Exception:
                pass
    reader = VideoReader(cache_root=CACHE, frame_extractor="decord", frame_resize=336)
    todo2 = [(k, r) for k, r in recs.items() if k not in picks]
    print(f"EXEC phase: {len(todo2)} to execute", flush=True)

    with open(picks_p, "a") as f:
        for i, (k, r) in enumerate(todo2):
            L = _distinct_letters(r)
            gold = str(r.get("answer_letter", "")).strip().upper()
            tt = r.get("task_type", "")
            if not L:
                pick, src, scores = "", "empty", {}
            else:
                maj = Counter(L).most_common(1)[0][0]
                distinct = sorted(set(L))
                if _is_gated(tt):
                    pick, src, scores = maj, "gated", {}
                elif len(distinct) == 1:
                    pick, src, scores = distinct[0], "single", {}
                elif k not in smap:
                    pick, src, scores = maj, "no_video", {}
                else:
                    snips = writes.get(k, {}).get("snippets", {})
                    vp = VideoPrimitives(smap[k]["video_path"], reader, n_pool=n_pool)
                    try:
                        vp.prewarm()   # decode + image-encode OUTSIDE the alarm (I/O not under timeout)
                    except Exception:
                        pass
                    prims = vp.as_namespace()
                    scores = {}
                    for d in distinct:
                        s, st = run_check(snips.get(d), prims, timeout=15) if snips.get(d) else (None, "empty")
                        if st == "ok":
                            scores[d] = s
                    if not scores:
                        pick, src = maj, "abstain_all"
                    else:
                        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                        top1L, top1 = ranked[0]
                        top2 = ranked[1][1] if len(ranked) > 1 else 0.0
                        if top1 > 0 and (top1 - top2) > _MARGIN:
                            pick, src = top1L, "exec"
                        else:
                            pick, src = maj, "abstain_band"
            f.write(json.dumps({"video_id": k[0], "question_id": k[1], "pick": pick,
                                "gold": gold, "src": src, "task_type": tt,
                                "scores": {kk: round(vv, 4) for kk, vv in scores.items()}}) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                print(f"  executed {i+1}/{len(todo2)}", flush=True)

    # ---------------------------------------------------------------- score
    allpicks = {}
    for line in open(picks_p):
        p = json.loads(line); allpicks[(p["video_id"], p["question_id"])] = p
    n = cp = cm = orc = ce = 0
    src_hist = Counter()
    by_tt = defaultdict(lambda: [0, 0, 0, 0])  # n, exec_correct, maj_correct, abstain
    for k, r in recs.items():
        p = allpicks.get(k)
        if not p:
            continue
        L = [str(t.get("prediction", "")).strip().upper() for t in r["trajectories"]
             if str(t.get("prediction", "")).strip()]
        if not L:
            continue
        g = str(r.get("answer_letter", "")).strip().upper()
        maj = Counter(L).most_common(1)[0][0]
        n += 1
        cp += L.count(g) / len(L)
        cm += int(maj == g)
        orc += int(g in set(L))
        ce += int(p["pick"] == g)
        src_hist[p["src"]] += 1
        b = by_tt[r.get("task_type", "?")]
        b[0] += 1; b[1] += int(p["pick"] == g); b[2] += int(maj == g)
        b[3] += int(p["src"] in ("gated", "abstain_all", "abstain_band", "no_video"))
    if n:
        print(f"\n=== EXECUTABLE VERIFIER (Mechanism B) — {exp.split('/')[-1]} / {bench} ===")
        print(f"n={n}  pass@1={cp/n:.4f}  majority={cm/n:.4f}  "
              f"EXEC_VERIFY={ce/n:.4f}  oracle={orc/n:.4f}")
        print(f"(reference: VMME-v2 frame-jury@8 = 0.4438)")
        print(f"headroom captured = {(ce/n - cp/n)/max(orc/n - cp/n, 1e-9)*100:.1f}%   "
              f"abstain/gate rate = {1 - src_hist['exec']/max(n,1):.2f}")
        print("src:", dict(src_hist))
        inv = "OK" if (orc/n + 1e-9 >= ce/n >= cp/n - 1e-9 and ce/n + 1e-9 >= cm/n) else "VIOLATED"
        print(f"invariant oracle>=exec>=pass1 and exec>=majority: {inv}")
        print("\nper task_type (n, exec_acc, maj_acc, exec-maj, abstain_frac):")
        for tt, b in sorted(by_tt.items(), key=lambda kv: -kv[1][0]):
            if b[0] >= 5:
                print(f"  {tt[:42]:42s} n={b[0]:3d} exec={b[1]/b[0]:.3f} maj={b[2]/b[0]:.3f} "
                      f"d={ (b[1]-b[2])/b[0]:+.3f} abst={b[3]/b[0]:.2f}")


if __name__ == "__main__":
    main()
