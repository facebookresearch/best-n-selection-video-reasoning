"""Direction 5 -- generative edit-a-frame verifier (CodeT-style counterfactual).

A GENERATIVE probe distinct from the failed CLIP-similarity oracle. For each distinct
candidate answer, instruction-edit a real frame toward the visual claim the answer
implies (FLUX.1-Kontext-dev). Intuition: if the answer is already TRUE of the video,
the frame barely needs to change (small edit); if FALSE, the editor must hallucinate
a large change. Score each candidate by EDIT MAGNITUDE (smaller = more consistent =
more supported); pick the smallest-edit candidate. Edit magnitude is measured two
ways (reported separately in the smoke so we can pick the discriminative one):
  - pix_l1  : mean |orig-edited| in [0,1] pixel space
  - clip_d  : 1 - cos(CLIP_img(orig), CLIP_img(edited))   (image-image, NOT text)

REQUIRES A GPU (FLUX inference). Run via SLURM/srun with venv-diffusers.
SMOKE-GATE: on ~10 questions, does the min-edit pick beat majority / track gold at
all? If the edit-magnitude signal does not separate correct from incorrect -> STOP
(report null; do not burn a full run). If it does -> full paired run vs frame-jury.

Usage: HF_HOME=... python verify_edit_frame.py <exp_dir> [--bench video-mme-v2]
  [--n-samples 500] [--frames 3] [--steps 20] [--limit N] [--model FLUX.1-Kontext-dev]
Writes picks to <exp_dir>/eval/<bench>/step_000000/edit_frame_picks.jsonl (resumable).
"""
from __future__ import annotations
import os, sys, re, json
from collections import Counter
from pathlib import Path

for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_v, None)
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

CODE = os.environ["VRITS_ROOT"] + "/experiments/109_muse_zeroshot_vmmev2_500/code"
CACHE = os.environ["VRITS_DATA"] + "/video_cache"
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.dirname(__file__))
import torch                                          # noqa: E402
from video_io import VideoReader                      # noqa: E402
from methods.vlm_backend import frames_to_pil         # noqa: E402
from benchmarks import load_benchmark                 # noqa: E402


def _argval(flag, default):
    return type(default)(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def _opt_text(options, letter):
    for o in options:
        s = str(o).strip()
        if s.upper().startswith(f"{letter}.") or s.upper().startswith(f"{letter})"):
            return re.sub(r"^[A-Ha-h][.)]\s*", "", s)
    return letter


def main():
    exp = sys.argv[1].rstrip("/")
    bench = _argval("--bench", "video-mme-v2")
    n_samples = _argval("--n-samples", 500)
    subset = _argval("--subset", "")
    n_frames = _argval("--frames", 3)     # candidate edit-target frames (uniform)
    steps = _argval("--steps", 20)
    limit = _argval("--limit", 10 ** 9)
    model = _argval("--model", "black-forest-labs/FLUX.1-Kontext-dev")

    import numpy as np
    from diffusers import FluxKontextPipeline
    from clip_primitives import encode_images         # image-image CLIP distance

    assert torch.cuda.is_available(), "verify_edit_frame requires a GPU (FLUX inference)"
    pipe = FluxKontextPipeline.from_pretrained(model, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    pred = sorted(Path(exp).glob("eval/**/predictions.jsonl"))[0]
    outp = pred.parent / "edit_frame_picks.jsonl"
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
                done.add(tuple(json.loads(line)[k] for k in ("video_id", "question_id")))
            except Exception:
                pass
    todo = [(k, r) for k, r in recs.items() if k not in done and k in smap][:limit]
    print(f"edit_frame: {len(recs)} recs, {len(done)} done, {len(todo)} to do", flush=True)
    reader = VideoReader(cache_root=CACHE, frame_extractor="decord", frame_resize=336)

    def edit_dist(orig, instr):
        """Mean edit magnitude of `orig` under instruction `instr`, two metrics."""
        out = pipe(image=orig, prompt=instr, guidance_scale=2.5,
                   num_inference_steps=steps, height=orig.height, width=orig.width).images[0]
        a = np.asarray(orig.convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0
        b = np.asarray(out.convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0
        pix = float(np.abs(a - b).mean())
        e = encode_images([orig, out])                # [2, D] normalized
        clip_d = float(1.0 - (e[0] @ e[1]).item())
        return pix, clip_d

    fout = open(outp, "a")
    for i, (k, r) in enumerate(todo):
        L = [str(t.get("prediction", "")).strip().upper() for t in r["trajectories"] if str(t.get("prediction", "")).strip()]
        gold = str(r.get("answer_letter", "")).strip().upper()
        maj = Counter(L).most_common(1)[0][0] if L else ""
        distinct = sorted(set(L))
        base = {"video_id": k[0], "question_id": k[1], "gold": gold, "maj": maj, "task_type": r.get("task_type", "")}
        if len(distinct) <= 1:
            fout.write(json.dumps({**base, "pick": (distinct[0] if distinct else ""), "src": "single"}) + "\n"); fout.flush(); continue
        s = smap[k]; q = r.get("meta", {}).get("question", ""); opts = r.get("meta", {}).get("options", [])
        try:
            frames, _ = reader.load_and_sample(s["video_path"], n_frames)
            imgs = frames_to_pil(frames)
        except Exception as e:
            fout.write(json.dumps({**base, "pick": maj, "src": f"decode_err"}) + "\n"); fout.flush(); continue
        pix_score, clip_score = {}, {}
        for d in distinct:
            instr = f"Edit the image so that it clearly shows: {_opt_text(opts, d)}"
            # average edit magnitude over the sampled target frames
            ps, cs = [], []
            for im in imgs:
                p, c = edit_dist(im, instr); ps.append(p); cs.append(c)
            pix_score[d] = sum(ps) / len(ps); clip_score[d] = sum(cs) / len(cs)
        # smaller edit = more consistent -> pick argmin distance (report both metrics)
        pick_pix = min(pix_score, key=pix_score.get)
        pick_clip = min(clip_score, key=clip_score.get)
        fout.write(json.dumps({**base, "pick": pick_clip, "pick_pix": pick_pix, "src": "edit",
                               "pix": {kk: round(vv, 4) for kk, vv in pix_score.items()},
                               "clip_d": {kk: round(vv, 4) for kk, vv in clip_score.items()}}) + "\n"); fout.flush()
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(todo)}", flush=True)
    fout.close()

    # score both metrics vs pass@1/majority/oracle
    picks = {tuple(json.loads(l)[k] for k in ("video_id", "question_id")): json.loads(l) for l in open(outp)}
    n = cm = orc = c_clip = c_pix = 0
    for k, r in recs.items():
        p = picks.get(k)
        if not p:
            continue
        L = [str(t.get("prediction", "")).strip().upper() for t in r["trajectories"] if str(t.get("prediction", "")).strip()]
        if not L:
            continue
        g = str(r.get("answer_letter", "")).strip().upper(); maj = Counter(L).most_common(1)[0][0]
        n += 1; cm += int(maj == g); orc += int(g in set(L))
        c_clip += int(p.get("pick") == g); c_pix += int(p.get("pick_pix", p.get("pick")) == g)
    if n:
        print(f"\n=== EDIT-A-FRAME (FLUX) -- {exp.split('/')[-1]} / {bench} ===")
        print(f"n={n}  majority={cm/n:.4f}  EDIT@clip={c_clip/n:.4f}  EDIT@pix={c_pix/n:.4f}  oracle={orc/n:.4f}")
        print(f"(reference: frame-jury@48=0.4879, @32=0.4438)  -- discriminative if EDIT >> majority")


if __name__ == "__main__":
    main()
