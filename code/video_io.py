"""Video reading and uniform frame sampling with on-disk caching.

Two backends are supported:
  - decord (fast; default)
  - pyav   (fallback used when decord fails)

Decoded frames are cached to disk as torch tensors keyed by (video_path, n_frames,
frame_resize) so that every method in the DAG shares the same cache and we never
re-decode the same clip twice across experiments.
"""

from __future__ import annotations

import os

# Strip broken HTTPS proxy before ANY networking-capable import (torch, decord's
# built-in tests etc.).
for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_v, None)

import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _cache_key(video_path: str, n_frames: int, frame_resize: Optional[int]) -> str:
    h = hashlib.sha1()
    h.update(video_path.encode("utf-8"))
    h.update(f"|n={n_frames}|r={frame_resize}".encode("utf-8"))
    return h.hexdigest()[:16]


def uniform_sample(n_total: int, n_frames: int) -> List[int]:
    """Return `n_frames` uniformly-spaced integer indices in [0, n_total).

    Raises ValueError when n_total <= 0. If n_total < n_frames the last frame is
    duplicated to reach length n_frames (deterministic).
    """
    if n_total <= 0:
        raise ValueError(f"cannot sample from a video with {n_total} frames")
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")
    if n_frames >= n_total:
        # Sample every frame and pad with the last one.
        idxs = list(range(n_total)) + [n_total - 1] * (n_frames - n_total)
        return idxs
    # np.linspace endpoint=False keeps indices strictly < n_total.
    idxs = np.linspace(0, n_total - 1, num=n_frames, dtype=np.int64)
    return idxs.tolist()


def _resize_short_side(frames: torch.Tensor, short_side: int) -> torch.Tensor:
    """Resize [T,H,W,C] uint8 tensor so the short side == `short_side`.

    Uses torch.nn.functional.interpolate on a NCHW float tensor, then casts back.
    """
    if short_side is None:
        return frames
    T, H, W, C = frames.shape
    if min(H, W) == short_side:
        return frames
    if H <= W:
        new_h = short_side
        new_w = max(1, int(round(W * short_side / H)))
    else:
        new_w = short_side
        new_h = max(1, int(round(H * short_side / W)))
    x = frames.permute(0, 3, 1, 2).float()
    x = torch.nn.functional.interpolate(
        x, size=(new_h, new_w), mode="bilinear", align_corners=False
    )
    x = x.clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
    return x


class VideoReader:
    """Load a video and expose uniform sampling with disk caching."""

    def __init__(
        self,
        cache_root: str,
        frame_extractor: str = "decord",
        frame_resize: Optional[int] = 336,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.frame_extractor = frame_extractor
        self.frame_resize = frame_resize

    # ------------------------------------------------------------------ decode
    def _decode_decord_idxs(self, path: str, idxs: List[int]) -> Tuple[torch.Tensor, float, int]:
        """Decode ONLY the requested indices from `path` via decord.

        Returns (frames [K,H,W,C] uint8, fps, n_total).
        """
        import decord

        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(path, num_threads=1)
        n_total = len(vr)
        fps = float(vr.get_avg_fps()) if vr.get_avg_fps() > 0 else 30.0
        # decord get_batch takes a list of frame indices — efficient random access,
        # no full decode.
        frames = vr.get_batch(idxs)
        return frames, fps, n_total

    def _decode_pyav_idxs(self, path: str, idxs: List[int]) -> Tuple[torch.Tensor, float, int]:
        """PyAV fallback that seeks to only the requested indices."""
        import av

        container = av.open(path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        n_total = int(stream.frames) if stream.frames else 0
        # Time base and duration for seeking.
        time_base = stream.time_base
        collected: dict = {}
        wanted = sorted(set(int(i) for i in idxs))
        for target_idx in wanted:
            # Convert frame index to PTS (approximate).
            target_time = target_idx / fps if fps > 0 else 0.0
            target_pts = int(target_time / float(time_base)) if time_base else 0
            try:
                container.seek(target_pts, stream=stream, any_frame=False, backward=True)
            except Exception:
                pass
            for j, frame in enumerate(container.decode(video=0)):
                if j > 60:  # safety cap: don't scan too far after a seek
                    break
                arr = frame.to_ndarray(format="rgb24")
                collected[target_idx] = torch.from_numpy(arr)
                break
        container.close()
        if not collected:
            raise RuntimeError(f"pyav decoded 0 frames from {path}")
        # Reorder according to `idxs` (may repeat).
        frames = torch.stack([collected[i] for i in idxs if i in collected], dim=0)
        return frames, fps, max(n_total, len(collected))

    def _decode_indices(self, path: str, idxs: List[int]) -> Tuple[torch.Tensor, float, int]:
        """Decode only the requested indices, decord first with pyav fallback."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"video not found: {path}")
        try:
            if self.frame_extractor == "decord":
                return self._decode_decord_idxs(path, idxs)
            return self._decode_pyav_idxs(path, idxs)
        except Exception as e:
            logger.warning(f"{self.frame_extractor} failed on {path}: {e!r}; falling back")
            if self.frame_extractor == "decord":
                return self._decode_pyav_idxs(path, idxs)
            return self._decode_decord_idxs(path, idxs)

    def _probe_n_total(self, path: str) -> Tuple[int, float]:
        """Fast probe of video length + fps without decoding frames."""
        try:
            import decord
            vr = decord.VideoReader(path, num_threads=1)
            n = len(vr)
            fps = float(vr.get_avg_fps()) if vr.get_avg_fps() > 0 else 30.0
            return n, fps
        except Exception:
            import av
            container = av.open(path)
            stream = container.streams.video[0]
            n = int(stream.frames) if stream.frames else 0
            fps = float(stream.average_rate) if stream.average_rate else 30.0
            container.close()
            return n, fps

    def load(self, path: str) -> Tuple[torch.Tensor, float]:
        """DEPRECATED full-video load — kept for compatibility. Do NOT use for
        long videos: it decodes every frame, which OOMs on 1080p clips >few sec.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"video not found: {path}")
        n_total, fps = self._probe_n_total(path)
        idxs = list(range(n_total))
        frames, _, _ = self._decode_indices(path, idxs)
        return frames, fps

    # ------------------------------------------------------------------- API
    def load_and_sample(
        self, path: str, n_frames: int
    ) -> Tuple[torch.Tensor, List[float]]:
        """Return (frames [n_frames,H,W,C] uint8, timestamps: seconds).

        Uses on-disk cache keyed by (path, n_frames, frame_resize). Cache format:
        {"frames": Tensor, "timestamps": List[float]} saved with torch.save.

        Decodes ONLY the sampled frame indices — never loads the full video.
        """
        key = _cache_key(path, n_frames, self.frame_resize)
        cache_path = self.cache_root / f"{key}.pt"
        if cache_path.exists():
            try:
                obj = torch.load(cache_path, map_location="cpu", weights_only=False)
                return obj["frames"], obj["timestamps"]
            except Exception as e:  # corrupt cache — regenerate
                logger.warning(f"cache read failed at {cache_path}: {e!r}; regenerating")

        # Probe first — cheap — then decode only the sampled indices.
        n_total, fps = self._probe_n_total(path)
        if n_total <= 0:
            raise RuntimeError(f"video has 0 frames per probe: {path}")
        idxs = uniform_sample(n_total, n_frames)
        frames, actual_fps, _ = self._decode_indices(path, idxs)
        if actual_fps > 0:
            fps = actual_fps
        frames = _resize_short_side(frames, self.frame_resize)
        timestamps = [float(i) / fps for i in idxs]

        # Atomic write: temp file then rename.
        tmp = cache_path.with_suffix(".pt.tmp")
        torch.save({"frames": frames, "timestamps": timestamps}, tmp)
        os.replace(tmp, cache_path)
        return frames, timestamps
