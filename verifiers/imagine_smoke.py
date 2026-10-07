"""imagine_loop smoke — co-residency (vLLM@0.55 + FLUX one process) + full answer() roundtrip.

Deterministic: forces a FLUX edit on a dummy frame (proves co-residency / no-OOM even
if the greedy model never chooses to IMAGINE), then runs the real Method.answer() on a
couple of real counterfactual/causal VMME-v2 samples. GPU-only; run under srun.
"""
import os, sys, time

for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_v, None)
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

CODE = os.environ["VRITS_ROOT"] + "/experiments/116_imagine_edit_27b_vmmev2_cf/code"
sys.path.insert(0, CODE)

import torch
from omegaconf import OmegaConf
from PIL import Image

from methods.vlm_backend import VLM
from methods import get_method_class
from video_io import VideoReader
from benchmarks import load_benchmark


def gb(x):
    return x / 1e9


def main():
    cfg = OmegaConf.load(CODE + "/config.yaml")
    print(f"[smoke] building VLM gpu_mem={cfg.model.gpu_memory_utilization} ...", flush=True)
    t = time.time()
    vlm = VLM(
        hf_id=str(cfg.model.hf_id), tp_size=1, dtype="bfloat16",
        max_model_len=int(cfg.model.max_model_len),
        gpu_memory_utilization=float(cfg.model.gpu_memory_utilization),
        thinking_mode=False, seed=42,
    )
    print(f"[smoke] VLM built in {time.time()-t:.0f}s | "
          f"cuda reserved={gb(torch.cuda.memory_reserved()):.1f}GB / 141GB", flush=True)

    reader = VideoReader(cache_root=str(cfg.video.cache_root),
                         frame_extractor="decord", frame_resize=336)
    Method = get_method_class("imagine_loop")
    method = Method(vlm=vlm, reader=reader, config=cfg)
    print(f"[smoke] method={method.name} mode={method.mode} "
          f"flux_steps={method.flux_steps} guidance={method.flux_guidance}", flush=True)

    # --- co-residency gate: force FLUX load + one edit on a dummy frame ---
    dummy = Image.new("RGB", (336, 336), (120, 120, 120))
    print("[smoke] loading FLUX + one edit (CO-RESIDENCY test) ...", flush=True)
    t = time.time()
    img = method._imagine(dummy, "a red apple on the table")
    torch.cuda.synchronize()
    print(f"[smoke] FLUX edit OK in {time.time()-t:.0f}s | out={img.size} | "
          f"cuda reserved={gb(torch.cuda.memory_reserved()):.1f}GB "
          f"max_reserved={gb(torch.cuda.max_memory_reserved()):.1f}GB / 141GB", flush=True)

    # --- verify vLLM still generates AFTER FLUX is resident ---
    txt, ntok = vlm.generate([dummy], "Describe this image in one word.", max_new_tokens=8)
    print(f"[smoke] vLLM post-FLUX generate OK: {txt!r} ({ntok} tok)", flush=True)

    # --- e2e: real answer() on 2 CF samples ---
    n = 0
    for s in load_benchmark("video-mme-v2", cache_root=str(cfg.video.cache_root),
                            task_types=list(cfg.benchmarks[0].task_types)):
        print(f"[smoke] === sample {s['video_id']} [{s.get('task_type')}] ===", flush=True)
        t = time.time()
        out = method.answer(question=s["question"], video_path=s["video_path"],
                            options=s["options"], subtitle=s.get("subtitle"),
                            benchmark="video-mme-v2")
        tools = [f"{c.get('type')}:{c.get('mode', c.get('letter', ''))}" for c in out["tool_calls"]]
        print(f"[smoke]   pred={out['prediction']!r} gold={s['answer_letter']} "
              f"n_tok={out['n_tokens']} n_frames={out['n_frames_used']} "
              f"tools={tools} time={time.time()-t:.0f}s", flush=True)
        n += 1
        if n >= 2:
            break
    print("[smoke] SMOKE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
