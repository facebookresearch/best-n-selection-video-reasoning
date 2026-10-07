# Zero-shot video-QA inference for Qwen3.5-{9B, 27B, 35B-A3B, 397B-A17B-FP8}

Shared inference infrastructure for the `video-reasoning-inference-time-scaling`
project. This code lives in the **root experiment** of the DAG and is inherited
by every child.

## Layout

```
code/
  config.yaml        # OmegaConf; every field is overridable on the CLI
  run.py             # entry point
  video_io.py        # decord/pyav decoder + cached uniform sampler
  benchmarks.py      # loaders for Video-MME-v2, Video-MME, LVBench, MLVU, EgoSchema
  prompts.py         # MC-QA prompt templates + robust A-H letter extraction
  eval.py            # per-benchmark scorers (incl. Video-MME-v2 Non-Lin Score)
  methods/
    __init__.py
    vlm_backend.py   # shared vLLM wrapper (thinking_mode aware)
    zeroshot_cot.py  # baseline
    jcef.py          # Just Caption Every Frame (Socratic)
    llovi.py         # dense captioning + multi-round summarize (Zhang 2023)
    videoagent.py    # iterative CLIP-retrieval agent (Wang 2024)
    vap.py           # Video Active Perception (Ma 2025, CLIP-anchored ablation)
  requirements.txt
```

## Config knobs

`config.yaml` uses OmegaConf, so every field is overridable via CLI:

```bash
python run.py model.hf_id=Qwen/Qwen3.5-27B model.tp_size=2 \
              inference.method=jcef inference.n_frames=32 \
              benchmarks.0.n_samples=100
```

Notes:

* `model.dtype=fp8` triggers vLLM `quantization=fp8` — used only for
  `Qwen3.5-397B-A17B-FP8`, which also needs `model.tp_size=8`.
* `model.thinking_mode=true` enables Qwen3.5 Think mode via
  `enable_thinking=True` in the chat template; `run.py` bumps
  `max_new_tokens` to 2048 in that case.
* Every method reads `inference.n_frames` (total budget) and its own optional
  section under `methods.{name}.*`.

## Outputs (relative to `code/`)

```
../results.json                                       # aggregated metrics per benchmark
../eval/{benchmark}/step_000000/predictions.jsonl     # one record per QA
../tb_logs/                                           # tensorboard scalars
```

`predictions.jsonl` is append-only and re-scanned on start, so preempted runs
resume seamlessly.

## Environment

* Python: `$PYTHON` (defaults to `python3`) — dependencies in `requirements.txt`
* Models cached at: `$HF_CACHE_HUB/models--Qwen--Qwen3.5-*` (defaults to `~/.cache/huggingface/hub`)
* Datasets cached at: `$HF_CACHE_HUB/datasets--*`

The HTTPS proxy on this cluster mangles TLS handshakes, so every file that does
network I/O starts with:

```python
for v in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'):
    os.environ.pop(v, None)
```

## Smoke test (10-sample debug run on Video-MME-v2)

Runs zeroshot_cot on Qwen3.5-9B with a 10-sample subset, TP=1, single H200:

```bash
cd "$VRITS_ROOT/code"
"${PYTHON:-python3}" run.py \
    model.hf_id=Qwen/Qwen3.5-9B \
    model.tp_size=1 \
    inference.method=zeroshot_cot \
    inference.n_frames=16 \
    'benchmarks=[{name: video-mme-v2, n_samples: 10}]'
```

Expected output: `../results.json` with a `video-mme-v2` block containing
`avg_acc`, `non_lin_score` (fallback to avg_acc if the released parquet lacks
group columns), and cost metrics (`n_frames_avg`, `tokens_avg`,
`time_per_query_avg`).

## Caveats and TODOs

* **MLVU downloads were still in progress at code-write
  time.** The loaders discover the annotation JSONs at runtime — if the
  snapshot is incomplete, they log a warning and skip the benchmark gracefully
  (they do NOT raise, so the run continues on the other benchmarks). Re-run
  once the datasets finish downloading.
* **LVBench videos ship as a separate zip** (`all_videos.zip` on the HF page)
  that may not be unpacked yet. `benchmarks.load_lvbench` searches
  `{snapshot}/all_videos/`, `{snapshot}/videos/`, and
  `{cache_root}/lvbench/` for `{key}.mp4`.
* **Video-MME-v2 non-linear scoring** is implemented as described in the
  paper (first-error truncation within reasoning-coherence groups; geometric
  mean within capability groups; arithmetic mean across capabilities). The
  released parquet exposes `group_type` and `group_structure`, which we treat
  as capability-group id and coherence-group index respectively. If the true
  scoring code disagrees with this convention we'll need to swap in the
  official implementation — the fallback branch (plain `avg_acc`) triggers
  automatically when `group_structure` is missing.
* **VAP** currently uses the paper's CLIP-anchored ablation (surprise =
  1 - CLIP(frame, video-summary-caption)); the CogVideoX-2B generation prior
  can be swapped into `methods/vap.py::_CLIPScorer` in a child experiment.
* **VideoAgent** requires `openai/clip-vit-large-patch14`; if it isn't cached,
  the retriever falls back to uniform sampling (logged warning).
