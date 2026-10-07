#!/bin/bash
# GPU launcher for the FLUX edit-a-frame verifier smoke (Direction 5).
for v in HTTP_PROXY HTTPS_PROXY http_proxy https_proxy HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; do unset $v; done
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_TOKEN=${HF_TOKEN:?set HF_TOKEN in your environment before running}
export DECORD_EOF_RETRY_MAX=20480
ROOT="${VRITS_ROOT:?set VRITS_ROOT to the repository root}"
exec "${PYTHON_DIFFUSERS:-python3}" \
  "$ROOT/verifiers/verify_edit_frame.py" \
  "$ROOT/experiments/109_bon8_verifier_zeroshot_27b_vmmev2" \
  --bench video-mme-v2 --frames 3 --steps 20 "$@"
