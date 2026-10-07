#!/bin/bash
# GPU launcher (TP=8) for the Qwen3.5-397B multimodal serving smoke.
for v in HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; do unset $v; done
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DECORD_EOF_RETRY_MAX=20480
export VLLM_ENABLE_V1_MULTIPROCESSING=1
ROOT="${VRITS_ROOT:?set VRITS_ROOT to the repository root}"
exec "${PYTHON:-python3}" "$ROOT/verifiers/serve397_smoke.py"
