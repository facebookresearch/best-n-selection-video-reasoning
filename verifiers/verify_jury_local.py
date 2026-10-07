"""Local large-Qwen holistic frame-jury (pushes the frame-budget win past Muse's 50-cap).

Same holistic N-way protocol as verify_jury_holistic.py (Muse @<=48 frames -> VMME
0.4879), but the jury is a LOCAL vLLM Qwen VLM (default Qwen3.5-397B-A17B-FP8, TP=4),
so #frames is unbounded by any API cap -> test 48 / 64 / 96. Two levers at once:
(a) a much LARGER jury model, (b) MORE frames.

Post-hoc over a stored BoN run's predictions.jsonl (trajectories). GPU job (vLLM).
NOTE: TP>1 uses spawn -> everything runs under `if __name__ == '__main__'`.

Usage (via srun, see jury_local.sh):
  python verify_jury_local.py <exp_dir> [--bench video-mme-v2] [--subset ""]
     [--n-samples 500] [--frames 48] [--model Qwen/Qwen3.5-397B-A17B-FP8]
     [--tp 4] [--dtype fp8] [--chunk 8] [--limit N]
Writes picks to <exp_dir>/eval/<bench>/step_000000/jury_local_<tag>_picks.jsonl (resumable).
"""
from __future__ import annotations
import os, sys, re, json
from collections import Counter
from pathlib import Path

for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_v, None)

CODE = os.environ["VRITS_ROOT"] + "/experiments/109_bon8_verifier_zeroshot_27b_vmmev2/code"
CACHE = os.environ["VRITS_DATA"] + "/video_cache"
sys.path.insert(0, CODE)

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
    # question_id may be None/"" (e.g. TempCompass: many questions per video) -> fall
    # back to the question text so rows don't collapse to one-per-video (mirrors the
    # holistic/gpt jury scorers; the local scorer was previously missing this fallback,
    # which silently collapsed TempCompass 1580 -> 410 unique videos).
    qid = question_id if question_id not in (None, "") else question
    return (video_id, qid)


def _prompt(rec, n_imgs):
    q = rec.get("meta", {}).get("question", ""); opts = rec.get("meta", {}).get("options", [])
    cand = "\n".join(
        f"Analysis {i+1} (concludes {t.get('prediction','?')}): {str(t.get('reasoning',''))[:500]}"
        for i, t in enumerate(rec["trajectories"])
    )
    return (
        f"You are shown {n_imgs} frames sampled uniformly from a video. Use the FRAMES as the "
        f"ground-truth evidence.\n\nQuestion: {q}\nOptions:\n" + "\n".join(opts) +
        f"\n\n{len(rec['trajectories'])} independent analyses of this video were produced:\n{cand}\n\n"
        "Verify the analyses against what you actually see in the frames and decide which option is "
        "correct. Respond with ONLY the final answer as 'Answer: X' (a single letter)."
    )


def main():
    exp = sys.argv[1].rstrip("/")
    bench = _argval("--bench", "video-mme-v2")
    n_samples = _argval("--n-samples", 500)
    subset = _argval("--subset", "")
    n_frames = _argval("--frames", 48)
    model = _argval("--model", "Qwen/Qwen3.5-397B-A17B-FP8")
    tp = _argval("--tp", 4)
    dtype = _argval("--dtype", "fp8")
    chunk = _argval("--chunk", 8)
    limit = _argval("--limit", 10 ** 9)
    max_model_len = _argval("--max-model-len", 49152)

    from methods.vlm_backend import VLM, frames_to_pil
    from video_io import VideoReader
    from benchmarks import load_benchmark

    tag = model.split("/")[-1].replace(".", "").replace("-", "")[:12] + f"_f{n_frames}"
    pred = sorted(Path(exp).glob("eval/**/predictions.jsonl"))[0]
    outp = pred.parent / f"jury_local_{tag}_picks.jsonl"

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
    print(f"jury_local[{tag}]: {len(recs)} recs, {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return

    vlm = VLM(hf_id=model, tp_size=tp, dtype=dtype, max_model_len=max_model_len,
              gpu_memory_utilization=0.92, seed=42,
              limit_mm_per_prompt={"image": n_frames + 4}, max_num_seqs=max(4, chunk))
    reader = VideoReader(cache_root=CACHE, frame_extractor="decord", frame_resize=336)

    fout = open(outp, "a")
    for c0 in range(0, len(todo), chunk):
        batch = []
        keys = []
        for k, r in todo[c0:c0 + chunk]:
            try:
                frames, _ = reader.load_and_sample(smap[k]["video_path"], n_frames)
                imgs = frames_to_pil(frames)
            except Exception:
                fout.write(json.dumps({"video_id": k[0], "question_id": k[1], "pick": "",
                                       "gold": str(r.get("answer_letter", "")).strip().upper(),
                                       "err": "decode"}) + "\n"); fout.flush()
                continue
            batch.append((imgs, _prompt(r, len(imgs)))); keys.append((k, r))
        if not batch:
            continue
        outs = vlm.generate_batch(batch=batch, max_new_tokens=1024, temperature=0.0)
        for (k, r), (txt, _n) in zip(keys, outs):
            fout.write(json.dumps({"video_id": k[0], "question_id": k[1], "pick": _extract(txt),
                                   "gold": str(r.get("answer_letter", "")).strip().upper()}) + "\n")
        fout.flush()
        print(f"  {min(c0+chunk, len(todo))}/{len(todo)}", flush=True)
    fout.close()

    # score
    picks = {}
    for line in open(outp):
        r = json.loads(line); picks[(r["video_id"], r["question_id"])] = r
    n = cj = cm = cp = orc = 0
    for k, r in recs.items():
        p = picks.get(k)
        if not p or not p.get("pick"):
            continue
        g = str(r.get("answer_letter", "")).strip().upper()
        L = [t["prediction"].strip().upper() for t in r["trajectories"] if str(t.get("prediction", "")).strip()]
        if not L:
            continue
        n += 1
        cj += int(p["pick"] == g); cm += int(Counter(L).most_common(1)[0][0] == g)
        cp += L.count(g) / len(L); orc += int(g in set(L))
    if n:
        print(f"\n=== LOCAL JURY {tag} -- {exp.split('/')[-1]} / {bench} ===")
        print(f"n={n}  pass@1={cp/n:.4f}  majority={cm/n:.4f}  LOCAL_JURY={cj/n:.4f}  oracle={orc/n:.4f}")
        print(f"(ref: Muse frame-jury@48 VMME=0.4879 Ego=0.776 ; oracle VMME=0.656 Ego=0.882)")


if __name__ == "__main__":
    main()
