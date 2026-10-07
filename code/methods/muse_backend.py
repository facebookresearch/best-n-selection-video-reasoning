"""MuseVLM — drop-in replacement for the vLLM `VLM` backend that calls Meta's
Model API (Muse Spark 1.1) via the public Responses API. No GPU: Muse runs
remotely; we only encode frames + orchestrate. Implements the same interface the
methods use:
  generate(images, text, ...) -> (text, n_tokens)
  generate_multiturn(turns, ...) -> (text, n_tokens)
  generate_batch(batch, ...) -> [(text, n_tokens), ...]

Transport (confirmed 2026-07-26, works from the login node, no GPU):
  POST https://api.meta.ai/v1/responses
  Authorization: Bearer $MODEL_API_KEY   (LaMa app token, format LLM|<appid>|<secret>)
  Content-Type: application/json
  body = {"model": "muse-spark-1.1-eval",
          "input": [ {"role": "user"|"assistant", "content": [parts...]} ],
          "max_output_tokens": N, "temperature": T}
    text  part = {"type": "input_text",  "text": ...}
    image part = {"type": "input_image", "image_url": "data:image/jpeg;base64,..."}
    assistant  = {"type": "output_text", "text": ...}   (for prior turns in `input`)
  answer text = concat of output[type==message].content[type==output_text].text
  n_tokens    = usage.output_tokens

Direct HTTPS: NO client cert, NO proxy (the localhost:10054 x2p proxy mangles
HTTPS-CONNECT). We strip proxy env vars at import. Muse is a *reasoning* model —
reasoning tokens count toward output_tokens, so max_output_tokens is floored to
_MIN_OUTPUT_TOKENS to guarantee the final answer message is emitted (avoids the
"all budget spent reasoning -> empty answer" truncation trap seen with Qwen-Think).
"""

from __future__ import annotations

import os

# The internal HTTPS proxy at localhost:10054 mangles HTTPS-CONNECT; api.meta.ai
# is reachable directly. Strip proxy vars so urllib goes direct.
for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_v, None)

import base64
import io
import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

_ENDPOINT = "https://api.meta.ai/v1/responses"
# Reasoning-model floor: reasoning tokens count toward output_tokens, so give
# enough headroom that the final answer message is always emitted even after a
# long chain of thought. Callers that ask for less (e.g. evloop rounds=2048,
# forced-answer=16) would otherwise truncate mid-reasoning -> empty prediction.
_MIN_OUTPUT_TOKENS = 32768


class MuseVLM:
    def __init__(self, model: str = "muse-spark-1.1-eval", max_img_side: int = 512,
                 jpeg_q: int = 70, seed: int = 42, thinking_mode: bool = False, **_) -> None:
        self.model = model
        key = os.environ.get("MODEL_API_KEY") or os.environ.get("MUSE_API_KEY")
        if not key:
            raise RuntimeError(
                "MODEL_API_KEY not set — export the Meta Model API LaMa token "
                "(format LLM|<appid>|<secret>) before running MuseVLM."
            )
        self.token = key
        self.max_img_side = max_img_side
        self.jpeg_q = jpeg_q
        self.seed = seed
        self.thinking_mode = thinking_mode
        self._preamble = (
            "Think through this very carefully and thoroughly, step by step, "
            "considering all the visual evidence, before you give your final answer.\n\n"
            if thinking_mode else ""
        )

    # -------------------------------------------------------------- images
    def _img_block(self, im: Image.Image) -> Dict[str, Any]:
        im = im.convert("RGB")
        w, h = im.size
        s = self.max_img_side
        if max(w, h) > s:
            if w >= h:
                im = im.resize((s, max(1, int(h * s / w))))
            else:
                im = im.resize((max(1, int(w * s / h)), s))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=self.jpeg_q)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return {"type": "input_image", "image_url": "data:image/jpeg;base64," + b64}

    # ---------------------------------------------------------------- call
    def _call(self, input_items: List[Dict[str, Any]], max_tokens: int,
              temperature: float) -> Tuple[str, int]:
        payload = {
            "model": self.model,
            "input": input_items,
            "max_output_tokens": max(int(max_tokens), _MIN_OUTPUT_TOKENS),
            "temperature": float(temperature),
        }
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            _ENDPOINT, data=body, method="POST",
            headers={"content-type": "application/json",
                     "authorization": "Bearer " + self.token})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    d = json.loads(r.read())
                txt = "".join(
                    c.get("text", "")
                    for o in d.get("output", []) if o.get("type") == "message"
                    for c in o.get("content", []) if c.get("type") == "output_text"
                )
                ntok = d.get("usage", {}).get("output_tokens", 0)
                # Reasoning-model truncation: if the model spent the whole budget
                # reasoning and never emitted the answer message (status incomplete,
                # empty text), continue via previous_response_id and force a concise
                # final answer. Token-efficient: only truncated samples pay extra, and
                # it guarantees an answer regardless of how long the reasoning ran.
                if (not txt) and d.get("status") == "incomplete" and d.get("id"):
                    try:
                        fu = {"model": self.model, "previous_response_id": d["id"],
                              "max_output_tokens": max(int(max_tokens), _MIN_OUTPUT_TOKENS),
                              "temperature": float(temperature),
                              "input": [{"role": "user", "content": [{"type": "input_text",
                                "text": "You were cut off before giving your final answer. "
                                        "Based on all your analysis so far, respond now with "
                                        "ONLY your final answer as 'Answer: X' (a single letter)."}]}]}
                        req2 = urllib.request.Request(
                            _ENDPOINT, data=json.dumps(fu).encode(), method="POST",
                            headers={"content-type": "application/json",
                                     "authorization": "Bearer " + self.token})
                        with urllib.request.urlopen(req2, timeout=300) as r2:
                            d2 = json.loads(r2.read())
                        txt2 = "".join(c.get("text", "") for o in d2.get("output", [])
                                       if o.get("type") == "message"
                                       for c in o.get("content", []) if c.get("type") == "output_text")
                        ntok += d2.get("usage", {}).get("output_tokens", 0)
                        if txt2:
                            txt = txt2
                    except Exception:
                        pass
                return txt, ntok
            except urllib.error.HTTPError as e:
                bodytxt = ""
                try:
                    bodytxt = e.read().decode()[:300]
                except Exception:
                    pass
                # 400 = malformed request; won't fix on retry.
                if e.code == 400:
                    logger.warning(f"muse HTTP 400: {bodytxt}")
                    return f"[api error 400: {bodytxt}]", 0
                if attempt == 4:
                    logger.warning(f"muse HTTP {e.code}: {bodytxt}")
                    return f"[api error {e.code}: {bodytxt}]", 0
                time.sleep(2 * (attempt + 1))
            except Exception as e:
                if attempt == 4:
                    return f"[api error: {e!r}]", 0
                time.sleep(2 * (attempt + 1))
        return "[api error]", 0

    # ------------------------------------------------------------- generate
    def generate(self, images: List[Image.Image], text: str, max_new_tokens: int = 512,
                 temperature: float = 0.0, top_p: float = 1.0) -> Tuple[str, int]:
        content: List[Dict[str, Any]] = [self._img_block(im) for im in images]
        content.append({"type": "input_text", "text": self._preamble + text})
        return self._call([{"role": "user", "content": content}], max_new_tokens, temperature)

    def generate_multiturn(self, turns: List[Dict[str, Any]], max_new_tokens: int = 512,
                           temperature: float = 0.0, top_p: float = 1.0) -> Tuple[str, int]:
        input_items: List[Dict[str, Any]] = []
        for i, t in enumerate(turns):
            role = t["role"]
            txt = t.get("text", "")
            if i == 0 and role == "user":
                txt = self._preamble + txt
            content: List[Dict[str, Any]] = []
            if role == "assistant":
                # Prior assistant turns carry text only (as output_text). Skip if
                # empty — an empty content list is rejected by the API (the exact
                # bug that broke the Claude evloop runs).
                if txt:
                    content.append({"type": "output_text", "text": txt})
                if not content:
                    continue
            else:
                for im in (t.get("images") or []):
                    content.append(self._img_block(im))
                if txt:
                    content.append({"type": "input_text", "text": txt})
                if not content:
                    content.append({"type": "input_text", "text": " "})
            input_items.append({"role": role, "content": content})
        return self._call(input_items, max_new_tokens, temperature)

    def generate_batch(self, batch: List[Tuple[List[Image.Image], str]], max_new_tokens: int = 128,
                       temperature: float = 0.0, top_p: float = 1.0) -> List[Tuple[str, int]]:
        # No server-side batch on the Responses API — issue sequential calls.
        return [self.generate(imgs, txt, max_new_tokens, temperature, top_p) for imgs, txt in batch]
