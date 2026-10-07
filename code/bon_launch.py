"""bon_launch.py -- env wrapper for run.py (offline HF + proxy/SSL strip).

run.py (the BoN generator) does not set the offline/proxy env; under
launch-experiment it inherits the session's broken SSL_CERT_FILE + proxy and dies
trying to reach the HF hub for the model. This wrapper sets the offline env
(mirroring verifiers/jury_local.sh) then invokes run.py with the local config.
"""
import os
import subprocess
import sys

for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
           "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
    os.environ.pop(_v, None)
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DECORD_EOF_RETRY_MAX"] = "20480"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"

PY = os.environ.get("PYTHON", "python3")
HERE = os.path.dirname(os.path.abspath(__file__))
rc = subprocess.run(
    [PY, f"{HERE}/run.py", "--config", f"{HERE}/config.yaml"],
    env=os.environ, check=False,
).returncode
print(f"[bon_launch] run.py rc={rc}", flush=True)
sys.exit(rc)
