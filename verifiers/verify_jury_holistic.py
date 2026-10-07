"""Holistic frame-grounded jury (Phase-1 protocol), configurable #frames.

Reproduces the Phase-1 frame-jury (Muse re-watches N frames + picks among the N
candidate reasoning traces in ONE N-way judgment; VMME 0.4438 @ 32 frames). Exposed
--frames so we can test the MORE-FRAMES strengthening axis (32 -> 64) with the
protocol held fixed. One clean variable vs the Phase-1 baseline: frame count.

Post-hoc over a stored BoN run's predictions.jsonl (trajectories). No GPU.
Usage: MODEL_API_KEY=... python verify_jury_holistic.py <exp_dir> [--bench video-mme-v2]
  [--n-samples 500] [--subset ""] [--frames 64] [--workers 6] [--limit N]
Writes picks to <exp_dir>/eval/<bench>/step_000000/jury_holistic{frames}_picks.jsonl (resumable).
"""
from __future__ import annotations
import os, sys, re, json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
           "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
    os.environ.pop(_v, None)  # SSL_CERT_FILE points to a missing cafile post-update -> breaks httpx/requests

CODE = os.environ["VRITS_ROOT"] + "/experiments/109_muse_zeroshot_vmmev2_500/code"
CACHE = os.environ["VRITS_DATA"] + "/video_cache"
sys.path.insert(0, CODE)
from methods.muse_backend import MuseVLM            # noqa: E402
from methods.vlm_backend import frames_to_pil       # noqa: E402
from video_io import VideoReader                     # noqa: E402
from benchmarks import load_benchmark                # noqa: E402

_ANS = re.compile(r"answer\s*[:\-]\s*\(?([A-Ha-h])\)?", re.IGNORECASE)


def _extract(t):
    if not t:
        return ""
    m = _ANS.search(t)
    if m:
        return m.group(1).upper()
    for ch in reversed(re.findall(r"\b([A-Ha-h])\b", t)):
        return ch.upper()
    return ""


def _argval(flag, default):
    return type(default)(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def _key(video_id, question_id, question):
    # question_id may be None (e.g. TempCompass: multiple questions per video) ->
    # fall back to question text so each question keys uniquely.
    qid = question_id if question_id not in (None, "") else question
    return (video_id, qid)


def main():
    exp = sys.argv[1].rstrip("/")
    bench = _argval("--bench", "video-mme-v2")
    n_samples = _argval("--n-samples", 500)
    subset = _argval("--subset", "")
    n_frames = _argval("--frames", 64)
    workers = _argval("--workers", 6)
    limit = _argval("--limit", 10 ** 9)
    n_cand = _argval("--n-cand", 999)   # truncate to first-N candidates (use 8 to match the N=8 headline)
    standalone = ("--standalone" in sys.argv)  # candidate-free SOLVER mode (Muse standalone -> SI = selector - standalone)

    pred = sorted(Path(exp).glob("eval/**/predictions.jsonl"))[0]
    outp = pred.parent / (f"jury_holistic{n_frames}" + ("_standalone" if standalone else "") + "_picks.jsonl")
    recs = {}
    for line in open(pred):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("trajectories"):
            recs[_key(r.get("video_id"), r.get("question_id"), (r.get("meta") or {}).get("question"))] = r

    smap = {}
    _bkw = {"subset": subset} if subset else {}
    for s in load_benchmark(bench, cache_root=CACHE, n_samples=n_samples, **_bkw):
        smap[_key(s["video_id"], s.get("meta", {}).get("question_id"), s.get("question"))] = s

    done = set()
    if outp.exists():
        for line in open(outp):
            try:
                r = json.loads(line); done.add((r["video_id"], r["question_id"]))
            except Exception:
                pass
    todo = [(k, r) for k, r in recs.items() if k not in done and k in smap][:limit]
    print(f"jury_holistic{n_frames}: {len(recs)} recs, {len(smap)} bench, {len(done)} done, {len(todo)} to do", flush=True)

    vlm = MuseVLM(model="muse-spark-1.1-eval", thinking_mode=False, seed=42)
    reader = VideoReader(cache_root=CACHE, frame_extractor="decord", frame_resize=336)

    def work(k, r):
        s = smap[k]
        try:
            frames, _ = reader.load_and_sample(s["video_path"], n_frames)
            imgs = frames_to_pil(frames)
        except Exception as e:
            return {"video_id": k[0], "question_id": k[1], "pick": "", "err": f"decode:{e!r}",
                    "gold": str(r.get("answer_letter", "")).strip().upper()}
        q = r.get("meta", {}).get("question", ""); opts = r.get("meta", {}).get("options", [])
        if standalone:
            prompt = (
                f"You are shown {len(imgs)} frames sampled uniformly from a video. Use the FRAMES as the "
                f"ground-truth evidence.\n\nQuestion: {q}\nOptions:\n" + "\n".join(opts) +
                "\n\nAnswer the question using ONLY what you see in the frames. Respond with ONLY the "
                "final answer as 'Answer: X' (a single letter)."
            )
        else:
            trajs = r["trajectories"][:n_cand]
            cand = "\n".join(
                f"Analysis {i+1} (concludes {t.get('prediction','?')}): {str(t.get('reasoning',''))[:500]}"
                for i, t in enumerate(trajs)
            )
            prompt = (
                f"You are shown {len(imgs)} frames sampled uniformly from a video. Use the FRAMES as the "
                f"ground-truth evidence.\n\nQuestion: {q}\nOptions:\n" + "\n".join(opts) +
                f"\n\n{len(trajs)} independent analyses of this video were produced:\n{cand}\n\n"
                "Verify the analyses against what you actually see in the frames and decide which option is "
                "correct. Respond with ONLY the final answer as 'Answer: X' (a single letter)."
            )
        txt, _ = vlm.generate(images=imgs, text=prompt, max_new_tokens=8192, temperature=0.0)
        return {"video_id": k[0], "question_id": k[1], "pick": _extract(txt),
                "gold": str(r.get("answer_letter", "")).strip().upper()}

    with open(outp, "a") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, k, r) for k, r in todo]
        for i, fut in enumerate(as_completed(futs)):
            f.write(json.dumps(fut.result()) + "\n"); f.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(todo)}", flush=True)

    picks = {}
    for line in open(outp):
        r = json.loads(line); picks[(r["video_id"], r["question_id"])] = r
    n = cj = cm = cp = orc = 0
    for k, r in recs.items():
        p = picks.get(k)
        if not p or not p.get("pick"):
            continue
        g = str(r.get("answer_letter", "")).strip().upper()
        L = [t["prediction"].strip().upper() for t in r["trajectories"][:n_cand] if str(t.get("prediction", "")).strip()]
        if not L:
            continue
        n += 1
        cj += int(p["pick"] == g); cm += int(Counter(L).most_common(1)[0][0] == g)
        cp += L.count(g) / len(L); orc += int(g in set(L))
    if n:
        _role = "STANDALONE" if standalone else "JURY"
        print(f"\n=== HOLISTIC {_role} @ {n_frames} frames -- {exp.split('/')[-1]} / {bench} ===")
        print(f"n={n}  pass@1={cp/n:.4f}  majority={cm/n:.4f}  HOLISTIC_{_role}={cj/n:.4f}  oracle={orc/n:.4f}")


if __name__ == "__main__":
    main()
