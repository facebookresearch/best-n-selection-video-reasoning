"""Evidence-Loop — iterative evidence-accumulation agent.

Fixes the core flaw of all prior tool methods: the answer stage must USE the
retrieved evidence together with the original frames AND the model's own prior
reasoning, in one continuous conversation.

Design (genuine multi-turn — each turn preserves everything before it):

  Turn 1 (user):      [N overview frames w/ progress bars] + question +
                      "reason; if you need a closer look request ZOOM(a,b),
                       else answer".
  Turn 1 (assistant): <model's reasoning + optional ZOOM(...) or Answer: X>
  Turn 2 (user):      [M detail frames from the requested window] +
                      "here is the detail you asked for; continue reasoning
                       using BOTH the overview and these frames".
  Turn 2 (assistant): <continued reasoning + optional ZOOM or Answer>
  ... up to max_rounds ...
  Final (user):       "You must now answer. Answer: X".

Contrast with prior failures:
  - VTimeCoT v2 (32%): two INDEPENDENT generate() calls — stage 2 rebuilt the
    prompt from scratch and discarded the model's stage-1 reasoning.
  - TAMA (14%): flattened ALL frames + text history into ONE user turn; the
    model could not bind frames to tool calls and the ReAct XML confused it.
  - Timeline (10%): discarded frames entirely, reasoned over text only.

Here the conversation grows: every prior turn (frames + reasoning) stays in
context, and new evidence is appended as a fresh user turn.
"""

from __future__ import annotations

import os

for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_v, None)

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig
from PIL import Image, ImageDraw

from prompts import extract_answer_letter
from video_io import VideoReader, _resize_short_side, uniform_sample

from .vlm_backend import VLM, frames_to_pil

logger = logging.getLogger(__name__)


# ZOOM(t_start, t_end) — accepts seconds or fractions in [0,1].
_ZOOM_RE = re.compile(r"ZOOM\s*\(\s*([0-9:.]+)\s*,\s*([0-9:.]+)\s*\)", re.IGNORECASE)
_ANSWER_RE = re.compile(r"answer\s*[:\-]\s*\(?([A-Ha-h])\)?", re.IGNORECASE)


def _parse_time_arg(s: str, duration_s: float) -> float:
    s = s.strip()
    if ":" in s:
        parts = [float(p) for p in s.split(":")]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return parts[0]
    try:
        v = float(s)
    except ValueError:
        return 0.0
    if 0.0 <= v <= 1.0 and duration_s > 2.0:
        return v * duration_s
    return v


def _draw_progress_bar(img: Image.Image, t_frac: float,
                       hs: Optional[float] = None, he: Optional[float] = None,
                       bar_frac: float = 0.06) -> Image.Image:
    out = img.copy().convert("RGB")
    W, H = out.size
    bar_h = max(6, int(H * bar_frac))
    y0 = H - bar_h
    d = ImageDraw.Draw(out)
    d.rectangle([0, y0, W, H], fill=(80, 80, 80))
    if hs is not None and he is not None and he > hs:
        d.rectangle([int(W * max(0, hs)), y0, int(W * min(1, he)), H], fill=(50, 120, 220))
    d.rectangle([0, y0, W - 1, H - 1], outline=(0, 0, 0), width=2)
    x = int(W * max(0.0, min(1.0, t_frac)))
    r = max(4, bar_h // 3)
    cy = y0 + bar_h // 2
    d.ellipse([x - r, cy - r, x + r, cy + r], fill=(220, 30, 30), outline=(0, 0, 0))
    return out


def _stamp(imgs: List[Image.Image], idxs: List[int], n_total: int,
           highlight: Optional[Tuple[float, float]] = None) -> List[Image.Image]:
    hs, he = (highlight if highlight else (None, None))
    return [_draw_progress_bar(im, i / max(n_total - 1, 1), hs, he)
            for im, i in zip(imgs, idxs)]


class Method:
    name = "evidence_loop"

    def __init__(self, vlm: VLM, reader: VideoReader, config: DictConfig) -> None:
        self.vlm = vlm
        self.reader = reader
        self.cfg = config
        m = config.get("methods", {}).get("evidence_loop", {})
        self.n_overview = int(m.get("n_overview_frames", 24))
        self.n_zoom = int(m.get("n_zoom_frames", 12))
        self.max_rounds = int(m.get("max_rounds", 3))  # up to 3 reasoning turns
        self.round_max_new_tokens = int(m.get("round_max_new_tokens", 512))
        self.temperature = float(config.inference.temperature)
        self.top_p = float(config.inference.top_p)

    def _decode_stamped(self, video_path: str, idxs: List[int], n_total: int,
                        highlight: Optional[Tuple[float, float]] = None) -> List[Image.Image]:
        tensor, _, _ = self.reader._decode_indices(video_path, idxs)
        tensor = _resize_short_side(tensor, self.reader.frame_resize)
        return _stamp(frames_to_pil(tensor), idxs, n_total, highlight)

    def _first_user_text(self, question: str, options: List[str],
                         subtitle: Optional[str], benchmark: str) -> str:
        head = (
            "You are watching a video shown as uniformly-sampled overview frames. "
            "Each frame has a progress bar at the bottom; the red dot marks that "
            "frame's position in the video.\n"
        )
        body = [head]
        if subtitle:
            s = subtitle.strip()
            body.append(f"Subtitles:\n{s[:4000] + (' ...' if len(s) > 4000 else '')}\n")
        body.append(f"Question: {question}")
        body.append("Options:\n" + "\n".join(options))
        body.append(
            "\nReason step by step. If a specific moment needs a closer look, "
            "request it on its own line as ZOOM(t_start, t_end) using fractions "
            "in [0,1] of the video duration (e.g. ZOOM(0.30, 0.45)). You may zoom "
            f"up to {self.max_rounds - 1} times. When you have enough evidence, "
            "give your final answer as 'Answer: X' (a single letter)."
        )
        return "\n".join(body)

    def answer(
        self,
        question: str,
        video_path: str,
        options: List[str],
        subtitle: Optional[str] = None,
        benchmark: str = "video-mme-v2",
    ) -> Dict[str, Any]:
        n_total, fps = self.reader._probe_n_total(video_path)
        duration_s = n_total / fps if fps > 0 else 0.0

        overview_idxs = uniform_sample(n_total, self.n_overview)
        overview_imgs = self._decode_stamped(video_path, list(overview_idxs), n_total)

        # Conversation turns accumulate; images + reasoning are never discarded.
        turns: List[Dict[str, Any]] = [{
            "role": "user",
            "images": overview_imgs,
            "text": self._first_user_text(question, options, subtitle, benchmark),
        }]

        n_tok_total = 0
        n_frames_seen = self.n_overview
        tool_calls: List[Dict[str, Any]] = []
        letter: Optional[str] = None
        full_reasoning: List[str] = []

        for round_i in range(self.max_rounds):
            is_last = (round_i == self.max_rounds - 1)
            resp, n_tok = self.vlm.generate_multiturn(
                turns=turns,
                max_new_tokens=self.round_max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            n_tok_total += n_tok
            full_reasoning.append(f"[round {round_i}]\n{resp}")
            # Record the assistant turn so its reasoning stays in context.
            turns.append({"role": "assistant", "images": [], "text": resp})

            letter = extract_answer_letter(resp) or letter
            am = _ANSWER_RE.search(resp)
            zm = _ZOOM_RE.search(resp)

            # Answer given (and not also asking to zoom) -> done.
            if am and not zm:
                tool_calls.append({"round": round_i, "type": "answer",
                                   "letter": am.group(1).upper()})
                break

            # On the last round without an answer, force one and stop.
            if is_last:
                if not am:
                    turns.append({
                        "role": "user", "images": [],
                        "text": "You must answer now. Reply with only 'Answer: X'.",
                    })
                    resp2, n_tok2 = self.vlm.generate_multiturn(
                        turns=turns, max_new_tokens=16,
                        temperature=self.temperature, top_p=self.top_p,
                    )
                    n_tok_total += n_tok2
                    full_reasoning.append(f"[forced answer]\n{resp2}")
                    letter = extract_answer_letter(resp2) or letter
                break

            # Zoom requested -> decode window, append as a NEW user turn.
            if zm:
                t_start = _parse_time_arg(zm.group(1), duration_s)
                t_end = _parse_time_arg(zm.group(2), duration_s)
                t_start = max(0.0, min(duration_s, t_start))
                t_end = max(0.0, min(duration_s, t_end))
                if t_end <= t_start:
                    t_end = min(duration_s, t_start + max(0.5, 0.05 * duration_s))
                i_start = int(round(t_start * fps))
                i_end = max(int(round(t_end * fps)), i_start + 1)
                zoom_local = uniform_sample(i_end - i_start, self.n_zoom)
                zoom_idxs = [i_start + i for i in zoom_local]
                hs = t_start / duration_s if duration_s > 0 else 0.0
                he = t_end / duration_s if duration_s > 0 else 1.0
                zoom_imgs = self._decode_stamped(video_path, zoom_idxs, n_total, (hs, he))
                n_frames_seen += len(zoom_imgs)
                tool_calls.append({"round": round_i, "type": "zoom",
                                   "t_start": round(t_start, 2), "t_end": round(t_end, 2),
                                   "n_frames": len(zoom_imgs)})
                turns.append({
                    "role": "user",
                    "images": zoom_imgs,
                    "text": (
                        f"Here are {len(zoom_imgs)} detail frames from the window you "
                        f"requested (t={t_start:.1f}s-{t_end:.1f}s), shaded blue on the "
                        "progress bar. Continue your reasoning using BOTH the original "
                        "overview frames and these detail frames. Zoom again if needed, "
                        "otherwise give 'Answer: X'."
                    ),
                })
            else:
                # No answer, no zoom — nudge toward a decision.
                turns.append({
                    "role": "user", "images": [],
                    "text": "Either request ZOOM(t_start, t_end) or give 'Answer: X'.",
                })

        return {
            "prediction": letter or "",
            "reasoning": "\n\n".join(full_reasoning),
            "n_tokens": n_tok_total,
            "n_frames_used": n_frames_seen,
            "tool_calls": tool_calls,
        }
