"""Perception primitives for the executable evidence verifier (Mechanism B).

Self-contained: loads openai/clip-vit-large-patch14 directly (the same model the
methods' `_CLIPRetriever` uses) and decodes frames via the project VideoReader.
These are the ONLY functions exposed to VLM-written check-code (see sandbox.py):

  clip(text)              -> float  # max cosine(frame, text) over uniform whole-video frames
  clip_at(text, t0, t1)   -> float  # same, restricted to a [t0,t1]-second window
  duration()              -> float  # video length in seconds

CLIP scoring is the "sound-er oracle": the code — not a fallible VLM — reads the
pixels. Whole-video max-pool is the default because the primary target (exp 109) is
a ZEROSHOT best-of-N run whose trajectories carry no timestamps; clip_at is for the
evidence_loop / branching arms whose reasoning references zoom windows.

PERF: the whole-video pool's IMAGE embeddings are encoded ONCE (prewarm) and cached,
so each clip(text) call is only a cheap text-encode + dot product. Frame decoding +
image encoding happen in prewarm() OUTSIDE the sandbox alarm, so the per-check
timeout guards only the (fast, bounded) VLM-written arithmetic — not slow I/O.

Requires the shared infra dir (109_muse_zeroshot_vmmev2_500/code) on sys.path for
`video_io` and `methods.vlm_backend` — the caller (verify_clip_checks.py) inserts it.
"""
from __future__ import annotations

import os

for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_v, None)

import threading
from typing import List, Optional

import torch
from PIL import Image

from video_io import VideoReader, uniform_sample
from methods.vlm_backend import frames_to_pil

_CLIP_ID = "openai/clip-vit-large-patch14"
_clip = {"model": None, "proc": None, "device": None}
_clip_lock = threading.Lock()  # CLIP forward is serialized (shared model state)


def get_clip(device: Optional[str] = None):
    if _clip["model"] is None:
        from transformers import CLIPModel, CLIPProcessor
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip["model"] = CLIPModel.from_pretrained(_CLIP_ID).to(device).eval()
        _clip["proc"] = CLIPProcessor.from_pretrained(_CLIP_ID)
        _clip["device"] = device
    return _clip["model"], _clip["proc"], _clip["device"]


@torch.no_grad()
def encode_images(images: List[Image.Image]) -> Optional[torch.Tensor]:
    """Normalized CLIP image embeddings [N, D] (cpu float)."""
    if not images:
        return None
    model, proc, device = get_clip()
    with _clip_lock:
        px = proc(images=images, return_tensors="pt").to(device)
        # transformers 5.13.1: get_image_features returns an output object, so use
        # the projection head directly (verified identical to model(**).image_embeds).
        f = model.visual_projection(model.vision_model(**px).pooler_output)
        f = f / f.norm(dim=-1, keepdim=True)
    return f.detach().float().cpu()


@torch.no_grad()
def encode_text(text: str) -> torch.Tensor:
    """Normalized CLIP text embedding [1, D] (cpu float)."""
    model, proc, device = get_clip()
    with _clip_lock:
        tk = proc(text=[str(text)], return_tensors="pt", padding=True, truncation=True).to(device)
        f = model.text_projection(model.text_model(**tk).pooler_output)
        f = f / f.norm(dim=-1, keepdim=True)
    return f.detach().float().cpu()


def _max_cos(img_embeds: Optional[torch.Tensor], text: str) -> float:
    if img_embeds is None or img_embeds.shape[0] == 0:
        return 0.0
    te = encode_text(text)                       # [1, D]
    sims = (img_embeds @ te.T).squeeze(-1)        # [N]
    return float(sims.max().item())


class VideoPrimitives:
    """Frame-caching + embedding-caching perception API bound to ONE video."""

    def __init__(self, video_path: str, reader: VideoReader, n_pool: int = 24) -> None:
        self.video_path = video_path
        self.reader = reader
        self.n_pool = n_pool
        self._pool_embeds: Optional[torch.Tensor] = None   # [N, D] whole-video image embeds
        self._warm = False
        self._dur: Optional[float] = None

    # ---- one-time warmup (decode + image-encode) — call OUTSIDE the sandbox alarm
    def prewarm(self) -> None:
        if self._warm:
            return
        try:
            frames, _ = self.reader.load_and_sample(self.video_path, self.n_pool)
            self._pool_embeds = encode_images(frames_to_pil(frames))
        except Exception:
            self._pool_embeds = None
        self.duration()
        self._warm = True

    # ---- primitives exposed to the sandbox (fast: cached image embeds) --------
    def duration(self) -> float:
        if self._dur is None:
            try:
                n_total, fps = self.reader._probe_n_total(self.video_path)
                self._dur = float(n_total) / float(fps) if fps else 0.0
            except Exception:
                self._dur = 0.0
        return self._dur

    def clip(self, text: str) -> float:
        if not self._warm:
            self.prewarm()
        return _max_cos(self._pool_embeds, str(text))

    def clip_at(self, text: str, t0: float, t1: float, n: int = 8) -> float:
        try:
            n_total, fps = self.reader._probe_n_total(self.video_path)
            if not fps or n_total <= 0:
                return self.clip(text)
            i0 = max(0, min(n_total - 1, int(round(float(t0) * fps))))
            i1 = max(0, min(n_total - 1, int(round(float(t1) * fps))))
            if i1 <= i0:
                i1 = min(n_total - 1, i0 + 1)
            idxs = [i0 + i for i in uniform_sample(i1 - i0, n)]
            frames, _, _ = self.reader._decode_indices(self.video_path, idxs)
            return _max_cos(encode_images(frames_to_pil(frames)), str(text))
        except Exception:
            return self.clip(text)

    def as_namespace(self) -> dict:
        return {"clip": self.clip, "clip_at": self.clip_at, "duration": self.duration}
