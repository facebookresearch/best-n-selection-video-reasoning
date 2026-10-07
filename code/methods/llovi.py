"""LLoVi (Zhang et al. 2023, arXiv 2312.17235) — dense clip captioning +
multi-round summarization.

For each of N uniform clips of `clip_seconds` seconds, we caption the clip
(fed as the frames sampled within the clip window). We then have the LLM
first summarize the noisy captions, then answer using the summary + question.
"""

from __future__ import annotations

import os

for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_v, None)

from typing import Any, Dict, List, Optional

import torch
from omegaconf import DictConfig

from prompts import (
    CLIP_CAPTION_PROMPT,
    build_qa_prompt,
    extract_answer_letter,
)
from video_io import VideoReader, uniform_sample

from .vlm_backend import VLM, frames_to_pil


class Method:
    name = "llovi"

    def __init__(self, vlm: VLM, reader: VideoReader, config: DictConfig) -> None:
        self.vlm = vlm
        self.reader = reader
        self.cfg = config
        self.n_frames = int(config.inference.n_frames)  # total frames budget
        self.temperature = float(config.inference.temperature)
        self.top_p = float(config.inference.top_p)
        method_cfg = config.get("methods", {}).get("llovi", {})
        self.clip_seconds = float(method_cfg.get("clip_seconds", 4))
        self.caption_max_new_tokens = int(method_cfg.get("caption_max_new_tokens", 60))
        self.summarize_max_new_tokens = int(method_cfg.get("summarize_max_new_tokens", 512))
        self.answer_max_new_tokens = int(method_cfg.get("answer_max_new_tokens", 256))
        # Frames per clip: we send a few frames per clip to the VLM captioner.
        self.frames_per_clip = 4

    def answer(
        self,
        question: str,
        video_path: str,
        options: List[str],
        subtitle: Optional[str] = None,
        benchmark: str = "video-mme-v2",
    ) -> Dict[str, Any]:
        # Load ALL frames (once) so we can slice clips at arbitrary boundaries.
        all_frames, fps = self.reader.load(video_path)
        n_total = int(all_frames.shape[0])
        duration = n_total / fps if fps > 0 else 0.0
        n_clips = max(1, min(self.n_frames // self.frames_per_clip,
                             int(round(duration / self.clip_seconds))))
        if n_clips < 1:
            n_clips = 1

        # Uniformly split the video into n_clips windows; caption each with
        # frames_per_clip sampled frames.
        window_len = n_total / n_clips
        batch = []
        clip_ts: List[float] = []
        for c in range(n_clips):
            start = int(round(c * window_len))
            end = int(round((c + 1) * window_len))
            if end <= start:
                end = start + 1
            local_idxs = uniform_sample(end - start, self.frames_per_clip)
            frame_idxs = [start + i for i in local_idxs]
            frames = all_frames[frame_idxs]
            # Downsize each frame to match the shared cache resolution.
            imgs = frames_to_pil(frames)
            batch.append((imgs, CLIP_CAPTION_PROMPT))
            clip_ts.append(start / fps if fps > 0 else float(c))

        caption_outs = self.vlm.generate_batch(
            batch,
            max_new_tokens=self.caption_max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        captions = [
            f"[t={ts:.1f}s] {t.strip()}"
            for (t, _), ts in zip(caption_outs, clip_ts)
        ]
        n_tok_captions = sum(n for _, n in caption_outs)

        # Round 1: summarize noisy captions.
        summarize_prompt = (
            "Below are captions of successive clips of a video. Some captions "
            "may be noisy or incorrect. Write a coherent 3-6 sentence summary "
            "of the video that reconciles them into a single narrative.\n\n"
            f"Captions:\n" + "\n".join(captions)
        )
        summary_text, n_tok_sum = self.vlm.generate(
            images=[],
            text=summarize_prompt,
            max_new_tokens=self.summarize_max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        # Round 2: answer using summary + question.
        answer_prompt = (
            f"Video summary:\n{summary_text.strip()}\n\n"
            f"Question: {question}\n"
            f"Options:\n" + "\n".join(options) + "\n\n"
            "Give your final answer as a single letter prefixed by 'Answer: '."
        )
        if subtitle:
            answer_prompt = f"Subtitles (compact):\n{subtitle[:2000]}\n\n" + answer_prompt

        answer_text, n_tok_ans = self.vlm.generate(
            images=[],
            text=answer_prompt,
            max_new_tokens=self.answer_max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        letter = extract_answer_letter(answer_text)
        return {
            "prediction": letter or "",
            "reasoning": f"[SUMMARY]\n{summary_text}\n[ANSWER]\n{answer_text}",
            "n_tokens": n_tok_captions + n_tok_sum + n_tok_ans,
            "n_frames_used": n_clips * self.frames_per_clip,
            "tool_calls": [
                {"type": "clip_caption", "n_clips": n_clips,
                 "frames_per_clip": self.frames_per_clip},
                {"type": "summarize"},
                {"type": "answer"},
            ],
        }
