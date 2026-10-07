"""Local runner for the Muse Spark 1.1 closed-model study (login node, no GPU).

Reuses the existing method classes (zeroshot_cot / evidence_loop / ...) unchanged
by swapping in MuseVLM (same interface). Decodes frames on CPU, calls the Meta
Model API (Responses API) per sample, parallelizes across samples with a thread
pool, resumes from predictions.jsonl, writes results.json.

Requires MODEL_API_KEY (LaMa token) in the environment.
Usage: python run_muse.py --config code/run_muse_config.yaml --workers 4
"""

from __future__ import annotations

import os
for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_v, None)

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from omegaconf import OmegaConf

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from benchmarks import load_benchmark
from eval import score as score_benchmark
from methods import get_method_class
from methods.muse_backend import MuseVLM
from video_io import VideoReader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)

    exp_root = (CODE / cfg.output.results_dir).resolve()
    out_dir = exp_root / "eval" / cfg.benchmarks[0].name / "step_000000"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "predictions.jsonl"

    # resume — only count a sample done if it has a real (non-empty) prediction.
    # Records written when a request errored/truncated (empty / "[api error]")
    # are retried on the next launch.
    done = {}
    if preds_path.exists():
        for line in open(preds_path):
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    pred = str(r.get("prediction", "")).strip()
                    if pred and not pred.lower().startswith("[api error"):
                        done[(r["video_id"], r.get("question_id"))] = r
                except json.JSONDecodeError:
                    pass
    print(f"resuming: {len(done)} already done", flush=True)

    vlm = MuseVLM(model=str(cfg.model.hf_id),
                  thinking_mode=bool(cfg.model.get("thinking_mode", False)),
                  seed=int(cfg.inference.seed))
    reader = VideoReader(cache_root=str(cfg.video.cache_root),
                         frame_extractor=str(cfg.video.frame_extractor),
                         frame_resize=int(cfg.video.frame_resize) if cfg.video.frame_resize else None)
    method = get_method_class(str(cfg.inference.method))(vlm=vlm, reader=reader, config=cfg)

    bc = cfg.benchmarks[0]
    kw = {}
    if "n_samples" in bc and bc.n_samples is not None:
        kw["n_samples"] = int(bc.n_samples)
    if "subset" in bc:
        kw["subset"] = str(bc.subset)
    if "with_subtitles" in bc and bc.with_subtitles:
        kw["include_subtitle"] = True
    samples = [s for s in load_benchmark(bc.name, cache_root=str(cfg.video.cache_root), **kw)
               if (s["video_id"], s.get("meta", {}).get("question_id")) not in done]
    print(f"to process: {len(samples)}", flush=True)

    def work(s):
        t0 = time.time()
        try:
            out = method.answer(question=s["question"], video_path=s["video_path"],
                                options=s["options"], subtitle=s.get("subtitle"), benchmark=bc.name)
        except Exception as e:
            out = {"prediction": "", "reasoning": f"ERROR: {e!r}", "n_tokens": 0,
                   "n_frames_used": 0, "tool_calls": [{"type": "error", "message": repr(e)}]}
        return {
            "video_id": s["video_id"], "question_id": s.get("meta", {}).get("question_id"),
            "prediction": out["prediction"], "answer_letter": s["answer_letter"],
            "task_type": s["task_type"], "reasoning": out["reasoning"],
            "n_tokens": out["n_tokens"], "n_frames_used": out["n_frames_used"],
            "time_seconds": time.time() - t0, "tool_calls": out.get("tool_calls", []),
            "meta": {**s.get("meta", {}), "options": s["options"], "question": s["question"]},
        }

    records = list(done.values())
    n0 = len(records)
    t_start = time.time()
    with open(preds_path, "a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, s) for s in samples]
        for i, fut in enumerate(as_completed(futs)):
            rec = fut.result()
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            if (i + 1) % 25 == 0:
                acc = sum(1 for r in records if (r["prediction"] or "").strip().upper() ==
                          (r["answer_letter"] or "").strip().upper()) / max(len(records), 1)
                print(f"  {n0+i+1}/{n0+len(samples)}  acc={acc:.4f}  elapsed={(time.time()-t_start)/60:.1f}m", flush=True)

    metrics = score_benchmark(bc.name, records)
    all_metrics = {"config": OmegaConf.to_container(cfg, resolve=True),
                   "benchmarks": {bc.name: metrics},
                   "total_wall_seconds": time.time() - t_start}
    with open(exp_root / "results.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"DONE acc={metrics.get('avg_acc')}  wrote {exp_root/'results.json'}", flush=True)


if __name__ == "__main__":
    main()
