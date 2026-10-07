"""Strengthened frame-grounded verifier -- Think-Then-Verify (per-candidate grounded).

Phase-1 frame-jury: Muse re-watches 32 frames and picks among the N candidates in a
single N-way judgment (VMME 0.4438, +6pt over text-jury). This v2 strengthens the
PROTOCOL (SAME 32 frames): for EACH distinct candidate answer, Muse independently
verifies it against the frames -- localize the decisive moment, then CONFIRM/REFUTE
with a confidence 1-5. Pick the strongest-CONFIRMED candidate (all-refute / tie ->
majority). One-variable change vs the Phase-1 jury: decomposed, evidence-localized
per-candidate verification instead of a single holistic pick (Think, Then Verify,
arXiv 2603.04977). Tests whether careful per-hypothesis scrutiny closes more of the
jury->oracle residual (Phase-1: jury 0.444 vs oracle 0.656).

Post-hoc over a stored BoN run's predictions.jsonl (trajectories). No GPU.
Usage: MODEL_API_KEY=... python verify_jury_grounded.py <exp_dir> [--bench video-mme-v2]
  [--n-samples 500] [--subset ""] [--frames 32] [--workers 8] [--limit N]
Writes picks to <exp_dir>/eval/<bench>/step_000000/jury_grounded_picks.jsonl (resumable).
"""
from __future__ import annotations
import os, sys, re, json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_v, None)

CODE = os.environ["VRITS_ROOT"] + "/experiments/109_muse_zeroshot_vmmev2_500/code"
CACHE = os.environ["VRITS_DATA"] + "/video_cache"
sys.path.insert(0, CODE)
from methods.muse_backend import MuseVLM            # noqa: E402
from methods.vlm_backend import frames_to_pil       # noqa: E402
from video_io import VideoReader                     # noqa: E402
from benchmarks import load_benchmark                # noqa: E402

_VERDICT = re.compile(r"verdict\s*[:\-]\s*(confirm|refute)", re.IGNORECASE)
_CONF = re.compile(r"confidence\s*[:\-]\s*([1-5])", re.IGNORECASE)


def _argval(flag, default):
    return type(default)(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def _opt_text(options, letter):
    for o in options:
        s = str(o).strip()
        if s[:2].upper().replace(")", ".") == f"{letter}." or s.upper().startswith(f"{letter}.") or s.upper().startswith(f"{letter})"):
            return s
    return letter


def _score_verdict(txt):
    """CONFIRM -> +conf, REFUTE -> -conf, unparsed -> 0.0 (abstain-ish)."""
    if not txt:
        return 0.0
    v = _VERDICT.search(txt)
    c = _CONF.search(txt)
    conf = int(c.group(1)) if c else 3
    if not v:
        return 0.0
    return float(conf) if v.group(1).lower() == "confirm" else -float(conf)


def main():
    exp = sys.argv[1].rstrip("/")
    bench = _argval("--bench", "video-mme-v2")
    n_samples = _argval("--n-samples", 500)
    subset = _argval("--subset", "")
    n_frames = _argval("--frames", 32)
    workers = _argval("--workers", 8)
    limit = _argval("--limit", 10 ** 9)

    pred = sorted(Path(exp).glob("eval/**/predictions.jsonl"))[0]
    outp = pred.parent / "jury_grounded_picks.jsonl"
    recs = {}
    for line in open(pred):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("trajectories"):
            recs[(r.get("video_id"), r.get("question_id"))] = r

    smap = {}
    _bkw = {"subset": subset} if subset else {}
    for s in load_benchmark(bench, cache_root=CACHE, n_samples=n_samples, **_bkw):
        smap[(s["video_id"], s.get("meta", {}).get("question_id"))] = s

    done = set()
    if outp.exists():
        for line in open(outp):
            try:
                r = json.loads(line); done.add((r["video_id"], r["question_id"]))
            except Exception:
                pass
    todo = [(k, r) for k, r in recs.items() if k not in done and k in smap][:limit]
    print(f"jury_grounded: {len(recs)} recs, {len(smap)} bench, {len(done)} done, {len(todo)} to do", flush=True)

    vlm = MuseVLM(model="muse-spark-1.1-eval", thinking_mode=False, seed=42)
    reader = VideoReader(cache_root=CACHE, frame_extractor="decord", frame_resize=336)

    def work(k, r):
        L = [str(t.get("prediction", "")).strip().upper() for t in r["trajectories"] if str(t.get("prediction", "")).strip()]
        gold = str(r.get("answer_letter", "")).strip().upper()
        maj = Counter(L).most_common(1)[0][0] if L else ""
        distinct = sorted(set(L))
        base = {"video_id": k[0], "question_id": k[1], "gold": gold, "maj": maj,
                "task_type": r.get("task_type", "")}
        if len(distinct) <= 1:
            return {**base, "pick": (distinct[0] if distinct else ""), "src": "single", "scores": {}}
        s = smap[k]
        try:
            frames, _ = reader.load_and_sample(s["video_path"], n_frames)
            imgs = frames_to_pil(frames)
        except Exception as e:
            return {**base, "pick": maj, "src": f"decode_err", "scores": {}}
        q = r.get("meta", {}).get("question", ""); opts = r.get("meta", {}).get("options", [])
        scores = {}
        for d in distinct:
            prompt = (
                f"You are shown {len(imgs)} frames sampled uniformly from a video. Use them as the "
                f"ground-truth evidence.\n\nQuestion: {q}\nProposed answer: {_opt_text(opts, d)}\n\n"
                "Examine the frames and identify the specific moment(s) that CONFIRM or REFUTE this "
                "proposed answer. Then respond in EXACTLY this format:\n"
                "VERDICT: CONFIRM or REFUTE\nCONFIDENCE: <an integer 1-5>"
            )
            txt, _ = vlm.generate(images=imgs, text=prompt, max_new_tokens=8192, temperature=0.0)
            scores[d] = _score_verdict(txt)
        # pick strongest-confirmed; require a positive verdict, else majority
        best = max(scores, key=scores.get)
        pick = best if scores[best] > 0 else maj
        src = "verified" if scores[best] > 0 else "all_refute_maj"
        return {**base, "pick": pick, "src": src, "scores": {kk: round(vv, 1) for kk, vv in scores.items()}}

    with open(outp, "a") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, k, r) for k, r in todo]
        for i, fut in enumerate(as_completed(futs)):
            f.write(json.dumps(fut.result()) + "\n"); f.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(todo)}", flush=True)

    # score grounded-jury@N vs pass@1/majority/oracle over all judged questions
    picks = {}
    for line in open(outp):
        r = json.loads(line); picks[(r["video_id"], r["question_id"])] = r
    n = cp = cm = orc = cj = 0
    src_hist = Counter()
    for k, r in recs.items():
        p = picks.get(k)
        if not p:
            continue
        L = [str(t.get("prediction", "")).strip().upper() for t in r["trajectories"] if str(t.get("prediction", "")).strip()]
        if not L:
            continue
        g = str(r.get("answer_letter", "")).strip().upper()
        maj = Counter(L).most_common(1)[0][0]
        n += 1
        cp += L.count(g) / len(L); cm += int(maj == g); orc += int(g in set(L)); cj += int(p["pick"] == g)
        src_hist[p["src"]] += 1
    if n:
        print(f"\n=== GROUNDED (Think-Then-Verify) JURY -- {exp.split('/')[-1]} / {bench} ===")
        print(f"n={n}  pass@1={cp/n:.4f}  majority={cm/n:.4f}  GROUNDED_JURY={cj/n:.4f}  oracle={orc/n:.4f}")
        print(f"(reference: Phase-1 N-way frame-jury VMME=0.4438)")
        print(f"headroom captured = {(cj/n-cp/n)/max(orc/n-cp/n,1e-9)*100:.1f}%   src={dict(src_hist)}")


if __name__ == "__main__":
    main()
