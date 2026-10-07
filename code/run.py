"""ProfAI experiment entry point for zeroshot video-QA inference.

Reads config.yaml (OmegaConf), instantiates the selected method, iterates over
each configured benchmark, writes predictions to
`../eval/{benchmark}/step_000000/predictions.jsonl`, and after everything is
done writes `../results.json` with aggregated metrics.

Progress is checkpointed every N samples so preempted runs can resume.
"""

from __future__ import annotations

import os

# Strip broken HTTPS proxy BEFORE any importer touches the network.
for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_v, None)

# vLLM external_launcher executor requires deterministic V1 (no worker
# multiprocessing) — must be set before any vLLM import.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Local imports (methods/, eval.py, prompts.py, video_io.py, benchmarks.py)
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from benchmarks import load_benchmark
from eval import score as score_benchmark
from methods import get_method_class
from methods.vlm_backend import VLM
from video_io import VideoReader


LOG = logging.getLogger("run")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handlers = [
        logging.StreamHandler(sys.stderr),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)


def _load_existing_predictions(path: Path) -> List[Dict[str, Any]]:
    """Return list of previously-written prediction records (for resume)."""
    if not path.exists():
        return []
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _pred_key(record: Dict[str, Any]) -> str:
    """Deterministic key used to detect already-answered questions."""
    m = record.get("meta") or {}
    qid = record.get("question_id") or m.get("question_id") or ""
    return f"{record.get('video_id')}|{qid}"


def _run_one_benchmark(
    bench_cfg: DictConfig,
    cfg: DictConfig,
    vlm: VLM,
    reader: VideoReader,
    method: Any,
    experiment_root: Path,
    tb_writer: Optional[Any],
) -> Dict[str, Any]:
    bench_name = bench_cfg.name
    bench_kwargs: Dict[str, Any] = {}
    if "split" in bench_cfg:
        bench_kwargs["split"] = bench_cfg.split
    if "n_samples" in bench_cfg and bench_cfg.n_samples is not None:
        bench_kwargs["n_samples"] = int(bench_cfg.n_samples)
    if "with_subtitles" in bench_cfg and bench_cfg.with_subtitles:
        bench_kwargs["include_subtitle"] = True

    out_dir = experiment_root / "eval" / bench_name / "step_000000"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "predictions.jsonl"

    existing = _load_existing_predictions(preds_path)
    done_keys = {_pred_key(r) for r in existing}
    if existing:
        LOG.info(f"[{bench_name}] resuming: {len(existing)} predictions already on disk")

    records: List[Dict[str, Any]] = list(existing)
    log_every = int(cfg.output.progress_log_every)
    step = len(existing)
    t0 = time.time()

    iterator = load_benchmark(bench_name, cache_root=cfg.video.cache_root, **bench_kwargs)
    # Use batched path if the method exposes answer_batch AND method_batch_size > 1.
    method_batch_size = int(getattr(cfg.inference, "batch_size", 32))
    has_batch = hasattr(method, "answer_batch") and method_batch_size > 1

    def _make_record(sample: Dict[str, Any], out: Dict[str, Any], elapsed: float) -> Dict[str, Any]:
        return {
            "video_id": sample["video_id"],
            "question_id": sample.get("meta", {}).get("question_id"),
            "prediction": out["prediction"],
            "answer_letter": sample["answer_letter"],
            "task_type": sample["task_type"],
            "reasoning": out["reasoning"],
            "n_tokens": out["n_tokens"],
            "n_frames_used": out["n_frames_used"],
            "time_seconds": elapsed,
            "tool_calls": out.get("tool_calls", []),
            "trajectories": out.get("trajectories", []),
            "meta": {
                **sample.get("meta", {}),
                "used_subtitle": sample.get("subtitle") is not None,
                "options": sample["options"],
                "question": sample["question"],
            },
        }

    def _log_step(record: Dict[str, Any]) -> None:
        nonlocal step
        step += 1
        if tb_writer is not None:
            correct = int(record["prediction"].strip().upper() == record["answer_letter"].strip().upper())
            tb_writer.add_scalar(f"{bench_name}/correct", correct, step)
            tb_writer.add_scalar(f"{bench_name}/n_frames_used", record["n_frames_used"], step)
            tb_writer.add_scalar(f"{bench_name}/n_tokens", record["n_tokens"], step)
            tb_writer.add_scalar(f"{bench_name}/time_seconds", record["time_seconds"], step)
        if step % log_every == 0:
            running = sum(
                1 for r in records if r["prediction"].strip().upper() == r["answer_letter"].strip().upper()
            ) / max(len(records), 1)
            LOG.info(f"[{bench_name}] step={step} running_acc={running:.4f} "
                     f"elapsed_min={(time.time()-t0)/60:.1f}")

    # Under external_launcher, all ranks run identical computation in lockstep
    # (vLLM produces same outputs on all ranks). Only rank 0 does file I/O.
    _rank0 = int(os.environ.get("RANK", "0")) == 0
    import contextlib
    _open_cm = open(preds_path, "a") if _rank0 else contextlib.nullcontext()
    with _open_cm as f_out:
        # A no-op sink for non-rank-0 processes.
        if not _rank0:
            class _NullFile:
                def write(self, *a, **k): return 0
                def flush(self): pass
            f_out = _NullFile()
        if has_batch:
            LOG.info(f"[{bench_name}] using batched inference: batch_size={method_batch_size}")
            pending: List[Dict[str, Any]] = []

            def _flush_batch(batch_samples: List[Dict[str, Any]]) -> None:
                if not batch_samples:
                    return
                t_start = time.time()
                try:
                    outs = method.answer_batch(batch_samples, benchmark=bench_name)
                except Exception as e:
                    LOG.exception(f"[{bench_name}] method.answer_batch failed on batch of {len(batch_samples)}: {e!r}")
                    outs = [
                        {
                            "prediction": "",
                            "reasoning": f"ERROR: {e!r}",
                            "n_tokens": 0,
                            "n_frames_used": 0,
                            "tool_calls": [{"type": "error", "message": repr(e)}],
                        }
                        for _ in batch_samples
                    ]
                elapsed = (time.time() - t_start) / max(len(batch_samples), 1)
                for s, out in zip(batch_samples, outs):
                    rec = _make_record(s, out, elapsed)
                    records.append(rec)
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    _log_step(rec)
                f_out.flush()

            for sample in tqdm(
                iterator, desc=f"[{bench_name}]",
                disable=not sys.stderr.isatty(),
            ):
                key = f"{sample['video_id']}|{sample.get('meta', {}).get('question_id') or ''}"
                if key in done_keys:
                    continue
                pending.append(sample)
                if len(pending) >= method_batch_size:
                    _flush_batch(pending)
                    pending = []
            _flush_batch(pending)
        else:
            for sample in tqdm(
                iterator, desc=f"[{bench_name}]",
                disable=not sys.stderr.isatty(),
            ):
                key = f"{sample['video_id']}|{sample.get('meta', {}).get('question_id') or ''}"
                if key in done_keys:
                    continue
                t_start = time.time()
                try:
                    out = method.answer(
                        question=sample["question"],
                        video_path=sample["video_path"],
                        options=sample["options"],
                        subtitle=sample.get("subtitle"),
                        benchmark=bench_name,
                    )
                except Exception as e:
                    LOG.exception(f"[{bench_name}] method.answer failed on {key}: {e!r}")
                    out = {
                        "prediction": "",
                        "reasoning": f"ERROR: {e!r}",
                        "n_tokens": 0,
                        "n_frames_used": 0,
                        "tool_calls": [{"type": "error", "message": repr(e)}],
                    }
                elapsed = time.time() - t_start
                rec = _make_record(sample, out, elapsed)
                records.append(rec)
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_out.flush()
                _log_step(rec)

    LOG.info(f"[{bench_name}] scoring {len(records)} predictions")
    metrics = score_benchmark(bench_name, records)
    LOG.info(f"[{bench_name}] metrics: {json.dumps({k: v for k, v in metrics.items() if not isinstance(v, dict)}, indent=2)}")
    if tb_writer is not None:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                tb_writer.add_scalar(f"{bench_name}/final/{k}", float(v), 0)
    return metrics


def main() -> None:
    # -------- torchrun coexistence via vLLM's `external_launcher` executor.
    # ProfAI's launcher wraps us in `torchrun --nproc_per_node=N` when
    # --gpus=N. For vLLM tensor parallelism we tell vLLM to REUSE this
    # existing torch.distributed group (each torchrun rank becomes a vLLM TP
    # worker). This avoids vLLM spawning its own workers which fails when the
    # container's loopback rendezvous is blocked (see earlier debug notes:
    # vLLM master_addr=127.0.0.1 hardcoded, TCPStore times out).
    _tr_rank = int(os.environ.get("RANK", "0"))
    _tr_world = int(os.environ.get("WORLD_SIZE", "1"))
    _is_dist = _tr_world > 1
    _is_rank0 = (_tr_rank == 0) or not _is_dist
    if _is_dist:
        print(f"[external_launcher] rank={_tr_rank}/{_tr_world}; will pass distributed_executor_backend='external_launcher' to vLLM")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(CODE_DIR / "config.yaml"))
    parser.add_argument(
        "overrides", nargs="*",
        help="OmegaConf dotted overrides, e.g. model.hf_id=... benchmarks.0.n_samples=10",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    _seed_everything(int(cfg.inference.seed))

    experiment_root = (CODE_DIR / cfg.output.results_dir).resolve()
    # Only rank 0 sets up logging to file — other ranks log to stdout only to
    # keep the log file uncorrupted.
    if _is_rank0:
        _setup_logging(experiment_root)
    else:
        # Non-rank-0 processes: minimal logging so their output doesn't pollute
        # the rank-0 log file. Use plain format (avoid the logging %(name) syntax
        # colliding with printf-style % interpolation of the rank number).
        logging.basicConfig(level=logging.WARNING, format=f"[rank{_tr_rank}] %(message)s", force=True)

    if _is_rank0:
        LOG.info(f"experiment_root = {experiment_root}")
        LOG.info(f"config =\n{OmegaConf.to_yaml(cfg)}")

    # Think mode uses a larger max_new_tokens by default.
    if bool(cfg.model.thinking_mode) and int(cfg.inference.max_new_tokens) < 2048:
        if _is_rank0:
            LOG.info("thinking_mode=True -> bumping max_new_tokens to 2048")
        cfg.inference.max_new_tokens = 2048

    # TensorBoard (rank 0 only).
    tb_writer = None
    if _is_rank0:
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter(log_dir=str(experiment_root / "tb_logs"))
        except Exception as e:
            LOG.warning(f"tensorboard unavailable: {e!r}")

    # Build shared VLM + video reader.
    # ALL ranks must call LLM() when using external_launcher — each rank
    # becomes a TP worker. vLLM broadcasts input from rank 0 to others.
    vlm = VLM(
        hf_id=str(cfg.model.hf_id),
        tp_size=int(cfg.model.tp_size),
        dtype=str(cfg.model.dtype),
        max_model_len=int(cfg.model.max_model_len),
        gpu_memory_utilization=float(cfg.model.gpu_memory_utilization),
        thinking_mode=bool(cfg.model.thinking_mode),
        seed=int(cfg.inference.seed),
        distributed_executor_backend=("external_launcher" if _is_dist else None),
    )
    reader = VideoReader(
        cache_root=str(cfg.video.cache_root),
        frame_extractor=str(cfg.video.frame_extractor),
        frame_resize=int(cfg.video.frame_resize) if cfg.video.frame_resize else None,
    )

    method_cls = get_method_class(str(cfg.inference.method))
    method = method_cls(vlm=vlm, reader=reader, config=cfg)
    LOG.info(f"instantiated method: {method.name}")

    all_metrics: Dict[str, Any] = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "benchmarks": {},
    }
    t_total = time.time()
    for bench_cfg in cfg.benchmarks:
        try:
            metrics = _run_one_benchmark(
                bench_cfg=bench_cfg,
                cfg=cfg,
                vlm=vlm,
                reader=reader,
                method=method,
                experiment_root=experiment_root,
                tb_writer=tb_writer,
            )
        except Exception as e:
            LOG.exception(f"benchmark {bench_cfg.name} failed: {e!r}")
            metrics = {"error": repr(e)}
        all_metrics["benchmarks"][bench_cfg.name] = metrics

    all_metrics["total_wall_seconds"] = time.time() - t_total

    results_path = experiment_root / "results.json"
    if _is_rank0:
        with open(results_path, "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        LOG.info(f"wrote {results_path}")

    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()


if __name__ == "__main__":
    main()
