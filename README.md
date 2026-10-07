# Video Reasoning at Inference Time: Best-of-N and Verifier Selection

Research code for a measurement study of **inference-time scaling** on
reasoning-centric video question answering: best-of-N (BoN) sampling with a
video-LM, paired with a range of **verifiers** (self-consistency, cross-family
juries, grounded/holistic juries, closed-model juries) used to select among the
N candidate answers.

This is a **release snapshot of research code**, not a maintained library. It is
the code that produced the numbers in the paper, published so that those numbers
can be re-derived. Expect research-grade ergonomics.

## License

Released under **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0
International) — see [`LICENSE`](LICENSE). Attribution required; **no commercial
use**. Benchmark datasets and model weights are **not** covered by this license
and remain under their own terms — see *Data* below.

## Layout

```
code/         Generation + scoring: benchmark loaders, BoN driver, methods,
              VLM backends, prompts, the two instrumented jury runners.
verifiers/    Standalone verifier / jury scripts and their smoke tests.
analysis/     Rescoring and effect-size scripts used for the reported tables.
figures/      Scripts that regenerate the paper figures.
```

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

vLLM and `decord` are the version-sensitive dependencies; the pinned floors in
`requirements.txt` are the versions the study ran on. The `diffusers`-based
verifier arm (frame editing / imagine checks) wants its **own** environment —
point `PYTHON_DIFFUSERS` at that interpreter rather than mixing it with vLLM.

## Environment variables

Paths are not hardcoded. Two variables are **required**; the rest have defaults.

| Variable | Required | Meaning |
|---|---|---|
| `VRITS_ROOT` | **yes** | Root of this checkout. Scripts resolve sibling code and artifact paths from it. |
| `VRITS_DATA` | **yes** | Data root. Benchmarks are expected at `$VRITS_DATA/benchmarks/...` and decoded frame caches at `$VRITS_DATA/video_cache/...`. |
| `HF_CACHE_HUB` | no | HuggingFace **hub** cache holding `models--*` / `datasets--*`. Default `~/.cache/huggingface/hub`. |
| `HF_HOME` | no | HuggingFace home. Default `~/.cache/huggingface`. Set with `setdefault`, so an existing value in your environment wins. |
| `PYTHON` | no | Interpreter used by the shell wrappers. Default `python3`. |
| `PYTHON_DIFFUSERS` | no | Interpreter for the `diffusers` verifier arm. Default `python3`. |
| `VRBENCH_VIDEO_DIR` | no | VRBench video directory. Default `$VRITS_DATA/vrbench_videos/v001_360p`. |
| `VIDEO_MME_DIR` | no | Legacy Video-MME-v1 tree. Only used by the Video-MME loaders, which the paper does not report; defaults to a nonexistent path. |
| `GPT_API_ENV_FILE` | no | `.env` file supplying credentials for the closed-model jury (`verifiers/verify_jury_gpt.py`). Never commit this file. |

```bash
export VRITS_ROOT="$PWD"
export VRITS_DATA="$HOME/vrits-data"
```

## Data

Five benchmarks are reported in the paper. All are third-party; obtain each from
its own source under its own license and terms.

| Benchmark | Rows used | Notes |
|---|---|---|
| TempCompass | 1,580 questions | Multiple-choice; variable option count (2-4). |
| EgoSchema | 500 questions (subset split) | 5-way multiple choice. |
| Video-Holmes | 1,837 questions | The main flagship pool for the BoN/verifier analysis. |
| MLVU-Test | 502 questions | Scored here as flat accuracy, **not** the official task-weighted metric. |
| VRBench | 1,000 questions (118 videos) | The loader flattens each record's `mcq` map, so "1,000 rows" means 1,000 **questions**, not 1,000 videos. |

Loaders live in `code/benchmarks.py` and read from `$VRITS_DATA` / the
HuggingFace hub cache. Frame decoding is cached under
`$VRITS_DATA/video_cache/<benchmark>/`.

`code/benchmarks.py` also still carries Video-MME / Video-MME-v2 / LVBench /
MLVU-dev loaders. Those are functional but **not** part of the reported results.

## Running

`code/run.py` takes a config plus OmegaConf dotted overrides:

```bash
cd "$VRITS_ROOT/code"
"${PYTHON:-python3}" run.py --config config.yaml \
  video.cache_root="$VRITS_DATA/video_cache" \
  benchmarks.0.name=video-holmes \
  inference.n_trajectories=8 \
  inference.n_frames=48
```

The `cache_root` default in the YAMLs is repo-relative (`data/video_cache`)
because `run.py` does not expand `$VAR` inside the config file — override it on
the command line (as above) or point a symlink at your data root.

Three configs ship: `config.yaml` (open-model BoN), `run_config.yaml`, and
`run_muse_config.yaml` (closed-model arm).

Available `inference.method` values, from `code/methods/__init__.py`:
`zeroshot_cot`, `jcef`, `llovi`, `videoagent`, `vap`, `tama`, `evidence_loop`,
`best_of_n`. BoN wraps a base method via `inference.base_method`.

### Verifiers

`verifiers/` holds the selection stage: local juries
(`verify_jury_local.py`), cross-family juries (`verify_jury_xfamily.py`),
grounded and holistic juries, and a closed-model jury. Each consumes a
`predictions.jsonl` produced by a BoN run and writes a `*_picks.jsonl`
alongside it.

**These scripts open their output in append mode.** Re-running one over an
existing `*_picks.jsonl` appends rather than replaces. Delete or move the old
file first.

### Analysis and figures

`analysis/` contains the rescoring passes (strict-vs-standard accuracy,
matched-support selection improvement, cross-run budget comparison).
`figures/*.py` regenerate the paper figures. Both read run artifacts, so they
require a completed set of runs.

## Scoring conventions

`code/eval.py` has benchmark-specific scorers for Video-MME, Video-MME-v2,
LVBench, MLVU-dev, and EgoSchema. The four remaining reported anchors
(TempCompass, MLVU-Test, VRBench, Video-Holmes) are registered to a **generic
letter-match scorer** (`score_mc_letter`): one row = one question, predicted
letter compared to gold letter, no task re-weighting. This is the same
convention as the reported pass@1 and majority baselines, and it is what the
paper's numbers use — but it is *not* the official harness for every one of
those benchmarks (MLVU in particular officially weights tasks). Reported
Video-Holmes numbers came from this same flat convention, computed by a separate
rescoring pass rather than in-line during generation.

## Not included

* **LongVideoBench** — loader, config entries, and cell registries were removed
  from this release for licensing/compliance reasons. No reported number depends
  on it. (Note: LongVideoBench is a *different* dataset from LVBench, which is
  retained.)
* **The closed-model Muse arm requires an internal endpoint.**
  `code/methods/muse_backend.py` targets a Meta-internal Model API and will not
  work outside Meta; its results are reported but not externally reproducible.
  Credentials come from the environment — none are bundled.
* **Run artifacts** — predictions, picks files, logs, and figures are not
  shipped. Regenerate them from the commands above.
* Some analysis scripts from the internal working tree were dropped as stale
  (they plotted benchmarks the final paper does not report).

## Citation

```bibtex
@misc{vrits2026,
  title  = {Video Reasoning at Inference Time: Best-of-N and Verifier Selection},
  author = {TBD},
  year   = {2026},
  note   = {Preprint. Author list to be finalized.}
}
```
