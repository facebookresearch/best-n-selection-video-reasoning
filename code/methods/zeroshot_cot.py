"""Zero-shot CoT baseline: uniformly sample N frames, single VLM forward with
a chain-of-thought prompt.
"""

from __future__ import annotations

import os

for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_v, None)

from typing import Any, Dict, List, Optional

from omegaconf import DictConfig

from prompts import build_qa_prompt, extract_answer_letter
from video_io import VideoReader

from .vlm_backend import VLM, frames_to_pil


class Method:
    name = "zeroshot_cot"

    def __init__(self, vlm: VLM, reader: VideoReader, config: DictConfig) -> None:
        self.vlm = vlm
        self.reader = reader
        self.cfg = config
        self.n_frames = int(config.inference.n_frames)
        self.max_new_tokens = int(config.inference.max_new_tokens)
        self.temperature = float(config.inference.temperature)
        self.top_p = float(config.inference.top_p)

    def answer(
        self,
        question: str,
        video_path: str,
        options: List[str],
        subtitle: Optional[str] = None,
        benchmark: str = "video-mme-v2",
    ) -> Dict[str, Any]:
        frames, _ts = self.reader.load_and_sample(video_path, self.n_frames)
        images = frames_to_pil(frames)
        prompt = build_qa_prompt(
            benchmark=benchmark,
            question=question,
            options=options,
            subtitle=subtitle,
            include_cot=True,
        )
        text, n_tok = self.vlm.generate(
            images=images,
            text=prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        letter = extract_answer_letter(text)
        return {
            "prediction": letter or "",
            "reasoning": text,
            "n_tokens": n_tok,
            "n_frames_used": len(images),
            "tool_calls": [],
        }

    def answer_batch(
        self,
        samples: List[Dict[str, Any]],
        benchmark: str = "video-mme-v2",
    ) -> List[Dict[str, Any]]:
        """Batched zero-shot CoT: decode all videos, run one batched vLLM call."""
        batch: List = []
        n_frames_each: List[int] = []
        for s in samples:
            frames, _ = self.reader.load_and_sample(s["video_path"], self.n_frames)
            images = frames_to_pil(frames)
            prompt = build_qa_prompt(
                benchmark=benchmark,
                question=s["question"],
                options=s["options"],
                subtitle=s.get("subtitle"),
                include_cot=True,
            )
            batch.append((images, prompt))
            n_frames_each.append(len(images))

        results = self.vlm.generate_batch(
            batch=batch,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        out: List[Dict[str, Any]] = []
        for (text, n_tok), n_frames in zip(results, n_frames_each):
            letter = extract_answer_letter(text)
            out.append(
                {
                    "prediction": letter or "",
                    "reasoning": text,
                    "n_tokens": n_tok,
                    "n_frames_used": n_frames,
                    "tool_calls": [],
                }
            )
        return out
