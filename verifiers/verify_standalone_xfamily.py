"""Standalone zero-shot accuracy of any vLLM VLM (model-agnostic, via llm.chat).

The judge answers the question DIRECTLY from frames+options — NO candidate analyses.
This measures each judge's own task capability, so we can report the SELECTION INCREMENT
(jury-as-selector accuracy MINUS this standalone accuracy) and de-confound the
"diversity > scale" claim from "the winning judge is just a stronger solver" (ICLR audit
gap #1). Also supports --black-screen (blank frames) as the frame-grounding control.

Reads the benchmark directly (same first-N as the BoN runs -> matched question set).
Usage:
  python verify_standalone_xfamily.py --bench video-mme-v2 --n-samples 500 --frames 32
     --model OpenGVLab/InternVL3-8B --tp 1 --dtype bfloat16 [--black-screen] [--limit N]
Writes standalone_<tag>.jsonl next to a per-model out dir + prints accuracy.
"""
from __future__ import annotations
import os, sys, re, json, base64, io
from pathlib import Path

for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_v, None)
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

CODE = os.environ["VRITS_ROOT"] + "/experiments/109_bon8_verifier_zeroshot_27b_vmmev2/code"
CACHE = os.environ["VRITS_DATA"] + "/video_cache"
OUT = os.environ["VRITS_ROOT"] + "/analysis/standalone_juries"
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


def _key(video_id, question_id, question):
    # null/"" question_id (e.g. TempCompass: many Qs per video) -> fall back to the
    # question text so resume/scoring don't collapse many questions to one-per-video.
    qid = question_id if question_id not in (None, "") else question
    return (video_id, qid)


def _b64(img, max_side, q):
    w, h = img.size
    s = max(w, h)
    if s > max_side:
        img = img.resize((max(1, int(w * max_side / s)), max(1, int(h * max_side / s))))
    b = io.BytesIO()
    img.convert("RGB").save(b, format="JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def main():
    bench = _argval("--bench", "video-mme-v2")
    n_samples = _argval("--n-samples", 500)
    subset = _argval("--subset", "")
    n_frames = _argval("--frames", 32)
    model = _argval("--model", "OpenGVLab/InternVL3-8B")
    tp = _argval("--tp", 1)
    dtype = _argval("--dtype", "bfloat16")
    chunk = _argval("--chunk", 4)
    limit = _argval("--limit", 10 ** 9)
    mml = _argval("--max-model-len", 32768)
    max_side = _argval("--max-img-side", 448)
    jq = _argval("--jpeg-q", 85)
    black = ("--black-screen" in sys.argv)

    from vllm import LLM, SamplingParams
    from PIL import Image
    from video_io import VideoReader
    from methods.vlm_backend import frames_to_pil
    from benchmarks import load_benchmark

    tag = model.split("/")[-1].replace(".", "").replace("-", "")[:14] + f"_{bench}_f{n_frames}" + ("_black" if black else "")
    Path(OUT).mkdir(parents=True, exist_ok=True)
    outp = Path(OUT) / f"standalone_{tag}.jsonl"

    samples = []
    _bkw = {"subset": subset} if subset else {}
    for s in load_benchmark(bench, cache_root=CACHE, n_samples=n_samples, **_bkw):
        samples.append(s)
    done = set()
    if outp.exists():
        for line in open(outp):
            try:
                r = json.loads(line); done.add(_key(r.get("video_id"), r.get("question_id"), r.get("question")))
            except Exception:
                pass
    todo = [s for s in samples if _key(s["video_id"], s.get("meta", {}).get("question_id"), s.get("question")) not in done][:limit]
    print(f"standalone[{tag}] model={model}: {len(samples)} samples, {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        _score(outp); return

    llm = LLM(model=model, tensor_parallel_size=tp, trust_remote_code=True,
              dtype=dtype, max_model_len=mml, gpu_memory_utilization=0.90,
              limit_mm_per_prompt={"image": n_frames + 2}, max_num_seqs=max(4, chunk), seed=42)
    sp = SamplingParams(temperature=0.0, max_tokens=1024, seed=42)
    reader = VideoReader(cache_root=CACHE, frame_extractor="decord", frame_resize=336)

    def _prompt(s, n):
        return (f"You are shown {n} frames sampled uniformly from a video. Use the FRAMES as the "
                f"ground-truth evidence.\n\nQuestion: {s['question']}\nOptions:\n" + "\n".join(s["options"]) +
                "\n\nAnswer the question using ONLY what you see in the frames. Respond with ONLY the "
                "final answer as 'Answer: X' (a single letter).")

    fout = open(outp, "a")
    for c0 in range(0, len(todo), chunk):
        convs, keys = [], []
        for s in todo[c0:c0 + chunk]:
            k = _key(s["video_id"], s.get("meta", {}).get("question_id"), s.get("question"))
            try:
                fr, _ = reader.load_and_sample(s["video_path"], n_frames)
                imgs = frames_to_pil(fr)
                if black:
                    imgs = [Image.new("RGB", im.size, (0, 0, 0)) for im in imgs]
            except Exception:
                fout.write(json.dumps({"video_id": k[0], "question_id": k[1], "pick": "",
                                       "gold": str(s.get("answer_letter", "")).strip().upper(), "err": "decode"}) + "\n")
                fout.flush(); continue
            content = [{"type": "image_url", "image_url": {"url": _b64(im, max_side, jq)}} for im in imgs]
            content.append({"type": "text", "text": _prompt(s, len(imgs))})
            convs.append([{"role": "user", "content": content}]); keys.append((k, s))
        if not convs:
            continue
        outs = llm.chat(convs, sampling_params=sp)
        for (k, s), o in zip(keys, outs):
            fout.write(json.dumps({"video_id": k[0], "question_id": k[1], "pick": _extract(o.outputs[0].text),
                                   "gold": str(s.get("answer_letter", "")).strip().upper()}) + "\n")
        fout.flush()
        print(f"  {min(c0+chunk, len(todo))}/{len(todo)}", flush=True)
    fout.close()
    _score(outp)


def _score(outp):
    seen = {}
    for line in open(outp):
        try:
            r = json.loads(line)
        except Exception:
            continue
        seen[(r.get("video_id"), r.get("question_id"))] = r  # dedup by resolved key; last write wins
    n = c = 0
    for r in seen.values():
        p = str(r.get("pick", "")).strip().upper(); g = str(r.get("gold", "")).strip().upper()
        if not p:
            continue
        n += 1; c += int(p == g)
    if n:
        print(f"\n=== STANDALONE {outp.stem} ===  n={n}  acc={c/n:.4f}", flush=True)


if __name__ == "__main__":
    main()
