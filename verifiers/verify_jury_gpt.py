"""Frontier frame-grounded jury via MetaGen GPT-5.6 (Sol/Terra) over the api.llama.com
OpenAI-compatible passthrough. Mirrors verify_jury_holistic.py (Muse) but swaps the
backend to the OpenAI client. Post-hoc over a stored BoN run's predictions.jsonl. No GPU.

Usage:
  source "$GPT_API_ENV_FILE"   # a local, uncommitted file that exports LLAMA_API_KEY
  python verify_jury_gpt.py <exp_dir> --model gpt-5-6-sol-genai-responses \
     [--bench video-mme-v2] [--n-samples 500] [--subset ""] [--frames 32] [--workers 4] [--limit N]
Writes picks to <exp_dir>/eval/<bench>/step_000000/jury_gpt_<slug>_<frames>_picks.jsonl (resumable).
"""
from __future__ import annotations
import os, sys, re, json, io, base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
           "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
    os.environ.pop(_v, None)  # SSL_CERT_FILE points to a missing cafile post-update -> breaks httpx/requests

CODE = os.environ["VRITS_ROOT"] + "/experiments/109_muse_zeroshot_vmmev2_500/code"
CACHE = os.environ["VRITS_DATA"] + "/video_cache"
BASE_URL = "https://api.llama.com/experimental/passthrough/openai/v1/"
sys.path.insert(0, CODE)
from methods.vlm_backend import frames_to_pil       # noqa: E402
from video_io import VideoReader                     # noqa: E402
from benchmarks import load_benchmark                # noqa: E402
from openai import OpenAI                             # noqa: E402

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


def _b64(img, max_side=448, q=85):
    w, h = img.size
    s = max(w, h)
    if s > max_side:
        img = img.resize((max(1, int(w * max_side / s)), max(1, int(h * max_side / s))))
    b = io.BytesIO()
    img.convert("RGB").save(b, format="JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def _key(video_id, question_id, question):
    # question_id may be None (e.g. TempCompass: multiple questions per video) ->
    # fall back to the question text so each question keys uniquely.
    qid = question_id if question_id not in (None, "") else question
    return (video_id, qid)


def main():
    exp = sys.argv[1].rstrip("/")
    model = _argval("--model", "gpt-5-6-sol-genai-responses")
    bench = _argval("--bench", "video-mme-v2")
    n_samples = _argval("--n-samples", 500)
    subset = _argval("--subset", "")
    n_frames = _argval("--frames", 32)
    workers = _argval("--workers", 4)
    limit = _argval("--limit", 10 ** 9)
    max_ct = _argval("--max-completion-tokens", 16384)  # reasoning model: use max_completion_tokens, floor high
    standalone = ("--standalone" in sys.argv)  # candidate-free SOLVER mode (for selection-increment SI = jury - standalone)

    slug = model.replace("gpt-5-6-", "").replace("-genai-responses", "").replace("-", "")[:8]
    key = os.environ.get("LLAMA_API_KEY")
    if not key:
        print("ERROR: set LLAMA_API_KEY (source gpt_assets/metagen.env)", flush=True); sys.exit(2)
    client = OpenAI(api_key=key, base_url=BASE_URL, timeout=600, max_retries=4)

    pred = sorted(Path(exp).glob("eval/**/predictions.jsonl"))[0]
    outp = pred.parent / (f"jury_gpt_{slug}_{n_frames}" + ("_standalone" if standalone else "") + "_picks.jsonl")
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
    print(f"jury_gpt[{slug}]@{n_frames}f: {len(recs)} recs, {len(smap)} bench, {len(done)} done, {len(todo)} to do", flush=True)

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
            cand = "\n".join(
                f"Analysis {i+1} (concludes {t.get('prediction','?')}): {str(t.get('reasoning',''))[:500]}"
                for i, t in enumerate(r["trajectories"])
            )
            prompt = (
                f"You are shown {len(imgs)} frames sampled uniformly from a video. Use the FRAMES as the "
                f"ground-truth evidence.\n\nQuestion: {q}\nOptions:\n" + "\n".join(opts) +
                f"\n\n{len(r['trajectories'])} independent analyses of this video were produced:\n{cand}\n\n"
                "Verify the analyses against what you actually see in the frames and decide which option is "
                "correct. Respond with ONLY the final answer as 'Answer: X' (a single letter)."
            )
        content = [{"type": "image_url", "image_url": {"url": _b64(im)}} for im in imgs]
        content.append({"type": "text", "text": prompt})
        try:
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": content}],
                max_completion_tokens=max_ct)
            txt = resp.choices[0].message.content or ""
        except Exception as e:
            return {"video_id": k[0], "question_id": k[1], "pick": "", "err": f"api:{str(e)[:160]}",
                    "gold": str(r.get("answer_letter", "")).strip().upper()}
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
        L = [t["prediction"].strip().upper() for t in r["trajectories"] if str(t.get("prediction", "")).strip()]
        if not L:
            continue
        n += 1
        cj += int(p["pick"] == g); cm += int(Counter(L).most_common(1)[0][0] == g)
        cp += L.count(g) / len(L); orc += int(g in set(L))
    if n:
        _role = "STANDALONE" if standalone else "JURY"
        print(f"\n=== GPT-5.6 {_role} [{slug}] @ {n_frames} frames -- {exp.split('/')[-1]} / {bench} ===")
        print(f"n={n}  pass@1={cp/n:.4f}  majority={cm/n:.4f}  GPT_{_role}={cj/n:.4f}  oracle={orc/n:.4f}", flush=True)


if __name__ == "__main__":
    main()
