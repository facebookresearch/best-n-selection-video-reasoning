"""JCEF (Just Caption Every Frame) — Socratic pipeline baseline.

Per-frame caption -> concatenate with timestamps -> LLM answers using
captions + question. Reproduces the MoReVQA baseline family that hit ~66.7 on
NExT-QA.
"""

from __future__ import annotations

import os

for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_v, None)

from typing import Any, Dict, List, Optional

from omegaconf import DictConfig

from prompts import (
    CAPTION_PROMPT,
    SUMMARY_PROMPT,
    build_qa_prompt,
    extract_answer_letter,
)
from video_io import VideoReader

from .vlm_backend import VLM, frames_to_pil


class Method:
    name = "jcef"

    def __init__(self, vlm: VLM, reader: VideoReader, config: DictConfig) -> None:
        self.vlm = vlm
        self.reader = reader
        self.cfg = config
        self.n_frames = int(config.inference.n_frames)
        self.temperature = float(config.inference.temperature)
        self.top_p = float(config.inference.top_p)
        method_cfg = config.get("methods", {}).get("jcef", {})
        self.caption_max_new_tokens = int(method_cfg.get("caption_max_new_tokens", 80))
        self.aggregator_max_new_tokens = int(
            method_cfg.get("aggregator_max_new_tokens", config.inference.max_new_tokens)
        )

    def _caption_frames(self, images: List[Any], timestamps: List[float]) -> List[str]:
        # Batched, one image per prompt.
        batch = [([img], CAPTION_PROMPT) for img in images]
        outs = self.vlm.generate_batch(
            batch,
            max_new_tokens=self.caption_max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        captions = []
        for (text, _), ts in zip(outs, timestamps):
            captions.append(f"[t={ts:.1f}s] {text.strip()}")
        return captions

    def answer(
        self,
        question: str,
        video_path: str,
        options: List[str],
        subtitle: Optional[str] = None,
        benchmark: str = "video-mme-v2",
    ) -> Dict[str, Any]:
        frames, timestamps = self.reader.load_and_sample(video_path, self.n_frames)
        images = frames_to_pil(frames)
        captions = self._caption_frames(images, timestamps)
        n_tok_captions = sum(len(c) // 4 for c in captions)  # rough token estimate

        aggregator_prompt = SUMMARY_PROMPT.format(
            captions="\n".join(captions),
            question=question,
            options="\n".join(options),
        )
        if subtitle:
            aggregator_prompt = f"Subtitles (compact):\n{subtitle[:2000]}\n\n" + aggregator_prompt

        answer_text, n_tok = self.vlm.generate(
            images=[],  # aggregator step is text-only
            text=aggregator_prompt,
            max_new_tokens=self.aggregator_max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        letter = extract_answer_letter(answer_text)
        return {
            "prediction": letter or "",
            "reasoning": answer_text,
            "n_tokens": n_tok + n_tok_captions,
            "n_frames_used": len(images),
            "tool_calls": [
                {"type": "frame_caption", "n_frames": len(images)},
                {"type": "text_aggregator"},
            ],
        }
