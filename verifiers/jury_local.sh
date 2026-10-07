#!/bin/bash
# GPU launcher for the local large-Qwen frame-jury (verify_jury_local.py). Pass args through.
for v in HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; do unset $v; done
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DECORD_EOF_RETRY_MAX=20480
export VLLM_ENABLE_V1_MULTIPROCESSING=1
ROOT="${VRITS_ROOT:?set VRITS_ROOT to the repository root}"
exec "${PYTHON:-python3}" "$ROOT/verifiers/verify_jury_local.py" "$@"
