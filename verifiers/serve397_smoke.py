"""Linchpin de-risk: can vLLM serve Qwen3.5-397B-A17B-FP8 MULTIMODALLY on H200s?

Loads the 397B MoE+fp8+vision model via the proven VLM backend (TP=4) and runs one
multimodal generate over 8 real video frames. If this prints a coherent description,
the 'larger Qwen frame-jury' direction is unblocked. If vLLM can't load the
qwen3_5_moe+fp8+vision combo, we pivot to Qwen3.5-35B-A3B / Qwen3.6-27B.

NOTE: TP>1 uses `spawn` -> the whole body MUST be under `if __name__ == '__main__'`
(else spawned workers re-import + re-instantiate the LLM -> recursive-spawn crash).
"""
import os
import sys, time

CODE = os.environ["VRITS_ROOT"] + "/experiments/109_bon8_verifier_zeroshot_27b_vmmev2/code"
CACHE = os.environ["VRITS_DATA"] + "/video_cache"
sys.path.insert(0, CODE)


def main():
    from methods.vlm_backend import VLM, frames_to_pil
    from video_io import VideoReader
    from benchmarks import load_benchmark

    t = time.time()
    vlm = VLM(hf_id="Qwen/Qwen3.5-397B-A17B-FP8", tp_size=4, dtype="fp8",
              max_model_len=32768, gpu_memory_utilization=0.92, seed=42)
    print(f"[397B] LOADED in {time.time()-t:.0f}s", flush=True)

    s = next(iter(load_benchmark("video-mme-v2", cache_root=CACHE, n_samples=1)))
    reader = VideoReader(cache_root=CACHE, frame_extractor="decord", frame_resize=336)
    frames, _ = reader.load_and_sample(s["video_path"], 8)
    imgs = frames_to_pil(frames)
    t = time.time()
    txt, ntok = vlm.generate(images=imgs, text="Briefly describe what happens across these 8 video frames.",
                             max_new_tokens=128, temperature=0.0)
    print(f"[397B] MULTIMODAL GEN ok in {time.time()-t:.0f}s, {ntok} tok:\n{txt[:500]}", flush=True)
    print("[397B] SMOKE PASS" if txt and not txt.startswith("[") else "[397B] SMOKE SUSPECT", flush=True)


if __name__ == "__main__":
    main()
