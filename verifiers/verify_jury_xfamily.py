"""Cross-family holistic frame-jury (model-agnostic via vLLM llm.chat()).

Same holistic N-way protocol as verify_jury_local.py, but uses vLLM's chat() API
with base64 image_url content so it works for ANY vLLM-supported VLM family
(InternVL, Phi-3.5-V, Qwen, ...) — not just Qwen's manual chat template. Powers the
capability-tiered CROSS-FAMILY jury ladder: does a jury WEAKER than the 27B generator
still beat majority? -> diversity (decorrelated perception), not verifier capability.

Post-hoc over a stored BoN run's predictions.jsonl. GPU job (vLLM). TP>1 -> __main__ guard.

Usage:
  python verify_jury_xfamily.py <exp_dir> [--bench video-mme-v2] [--n-samples 500]
     [--frames 48] [--model OpenGVLab/InternVL3-38B] [--tp 1] [--dtype bfloat16]
     [--chunk 8] [--limit N] [--max-model-len 32768] [--max-img-side 448] [--jpeg-q 85]
Writes picks to <exp_dir>/eval/<bench>/step_000000/jury_xf_<tag>_picks.jsonl (resumable).
"""
from __future__ import annotations
import os, sys, re, json, base64, io
from collections import Counter
from pathlib import Path

for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_v, None)
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

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


def _argval(flag, d):
    return type(d)(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else d


def _b64(img, max_side, q):
    from PIL import Image  # noqa: F401
    w, h = img.size
    s = max(w, h)
    if s > max_side:
        img = img.resize((max(1, int(w * max_side / s)), max(1, int(h * max_side / s))))
    b = io.BytesIO()
    img.convert("RGB").save(b, format="JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def _prompt(rec, n):
    q = rec.get("meta", {}).get("question", "")
    opts = rec.get("meta", {}).get("options", [])
    cand = "\n".join(
        f"Analysis {i+1} (concludes {t.get('prediction','?')}): {str(t.get('reasoning',''))[:500]}"
        for i, t in enumerate(rec["trajectories"])
    )
    return (
        f"You are shown {n} frames sampled uniformly from a video. Use the FRAMES as the "
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
    model = _argval("--model", "OpenGVLab/InternVL3-38B")
    tp = _argval("--tp", 1)
    dtype = _argval("--dtype", "bfloat16")
    chunk = _argval("--chunk", 8)
    limit = _argval("--limit", 10 ** 9)
    mml = _argval("--max-model-len", 32768)
    max_side = _argval("--max-img-side", 448)
    jq = _argval("--jpeg-q", 85)
    mdp = _argval("--max-dynamic-patch", 1)  # InternVL tiles/frame (1 = thumbnail; caps tokens for many-frame video)

    from vllm import LLM, SamplingParams
    from video_io import VideoReader
    from methods.vlm_backend import frames_to_pil

    from benchmarks import load_benchmark

    tag = model.split("/")[-1].replace(".", "").replace("-", "")[:14] + f"_f{n_frames}"
    pred = sorted(Path(exp).glob("eval/**/predictions.jsonl"))[0]
    outp = pred.parent / f"jury_xf_{tag}_picks.jsonl"

    recs = {}
    for line in open(pred):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("trajectories"):
            _qid = r.get("question_id"); _qid = _qid if _qid not in (None, "") else (r.get("meta") or {}).get("question")
            recs[(r.get("video_id"), _qid)] = r
    smap = {}
    _bkw = {"subset": subset} if subset else {}
    for s in load_benchmark(bench, cache_root=CACHE, n_samples=n_samples, **_bkw):
        _sqid = s.get("meta", {}).get("question_id"); _sqid = _sqid if _sqid not in (None, "") else s.get("question")
        smap[(s["video_id"], _sqid)] = s
    done = set()
    if outp.exists():
        for line in open(outp):
            try:
                r = json.loads(line); done.add((r["video_id"], r["question_id"]))
            except Exception:
                pass
    todo = [(k, r) for k, r in recs.items() if k not in done and k in smap][:limit]
    print(f"jury_xf[{tag}] model={model}: {len(recs)} recs, {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return

    # NOTE: InternVL dynamic tiling is capped via config.json (max_dynamic_patch=1) rather than
    # mm_processor_kwargs — passing max_dynamic_patch as a kwarg crashes (transformers forwards it
    # to InternVLVideoProcessor.__init__ which rejects it). config edit => ~1 tile (~256 tok)/frame.
    _ = mdp  # retained for CLI compatibility; tiling now set in config
    llm = LLM(model=model, tensor_parallel_size=tp, trust_remote_code=True,
              dtype=dtype, max_model_len=mml, gpu_memory_utilization=0.90,
              limit_mm_per_prompt={"image": n_frames + 2}, max_num_seqs=max(4, chunk), seed=42)
    sp = SamplingParams(temperature=0.0, max_tokens=1024, seed=42)
    reader = VideoReader(cache_root=CACHE, frame_extractor="decord", frame_resize=336)

    fout = open(outp, "a")
    for c0 in range(0, len(todo), chunk):
        convs = []
        keys = []
        for k, r in todo[c0:c0 + chunk]:
            try:
                fr, _ = reader.load_and_sample(smap[k]["video_path"], n_frames)
                imgs = frames_to_pil(fr)
            except Exception:
                fout.write(json.dumps({"video_id": k[0], "question_id": k[1], "pick": "",
                                       "gold": str(r.get("answer_letter", "")).strip().upper(),
                                       "err": "decode"}) + "\n"); fout.flush(); continue
            content = [{"type": "image_url", "image_url": {"url": _b64(im, max_side, jq)}} for im in imgs]
            content.append({"type": "text", "text": _prompt(r, len(imgs))})
            convs.append([{"role": "user", "content": content}]); keys.append((k, r))
        if not convs:
            continue
        outs = llm.chat(convs, sampling_params=sp)
        for (k, r), o in zip(keys, outs):
            txt = o.outputs[0].text
            fout.write(json.dumps({"video_id": k[0], "question_id": k[1], "pick": _extract(txt),
                                   "gold": str(r.get("answer_letter", "")).strip().upper()}) + "\n")
        fout.flush()
        print(f"  {min(c0+chunk, len(todo))}/{len(todo)}", flush=True)
    fout.close()

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
        print(f"\n=== XFAMILY JURY {tag} model={model} -- {exp.split('/')[-1]} / {bench} ===")
        print(f"n={n}  pass@1={cp/n:.4f}  majority={cm/n:.4f}  XF_JURY={cj/n:.4f}  oracle={orc/n:.4f}")
        print("(ref: Muse cross-family@48 VMME=0.4879 ; same-family 397B@48=0.356 ; majority=0.365)")


if __name__ == "__main__":
    main()
