"""VAP (Video Active Perception) — Ma et al. 2025, arXiv 2605.01662.

Training-free frame selection using a "surprise" prior. For the baseline
reproduction (no generative prior model) we use the paper's ablation setting:

    surprise(frame) = 1 - CLIP(frame, video_summary_caption)

Pipeline:
  1. Uniformly sample `candidate_frames` frames from the video.
  2. VLM generates a one-sentence video summary from a coarse uniform pass
     (init 8 frames).
  3. CLIP scores each candidate frame vs. the summary; surprise = 1 - sim.
  4. Take the top-`select_k` candidates by surprise.
  5. VLM answers the question conditioned on selected frames.

If real generation priors (e.g. CogVideoX-2B) are wired in later, only step 3
needs to change.
"""

from __future__ import annotations

import os

for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_v, None)

import logging
from typing import Any, Dict, List, Optional

import torch
from omegaconf import DictConfig
from PIL import Image

from prompts import (
    VIDEO_SUMMARY_ANCHOR_PROMPT,
    build_qa_prompt,
    extract_answer_letter,
)
from video_io import VideoReader, uniform_sample

from .vlm_backend import VLM, frames_to_pil

logger = logging.getLogger(__name__)


class _CLIPScorer:
    def __init__(self, device: str = "cuda") -> None:
        from transformers import CLIPModel, CLIPProcessor

        self._device = device
        model_id = "openai/clip-vit-large-patch14"
        try:
            self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
            self.processor = CLIPProcessor.from_pretrained(model_id)
            self.enabled = True
        except Exception as e:
            logger.warning(f"CLIP unavailable ({e!r}); VAP will fall back to uniform selection")
            self.enabled = False

    @torch.no_grad()
    def similarities(self, images: List[Image.Image], text: str) -> torch.Tensor:
        assert self.enabled
        inputs = self.processor(
            text=[text], images=images, return_tensors="pt", padding=True
        ).to(self._device)
        out = self.model(**inputs)
        image_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        text_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return (image_emb @ text_emb.T).squeeze(-1).cpu()


class Method:
    name = "vap"

    def __init__(self, vlm: VLM, reader: VideoReader, config: DictConfig) -> None:
        self.vlm = vlm
        self.reader = reader
        self.cfg = config
        method_cfg = config.get("methods", {}).get("vap", {})
        self.candidate_frames = int(method_cfg.get("candidate_frames", 32))
        self.select_k = int(method_cfg.get("select_k", 8))
        self.summary_max_new_tokens = int(method_cfg.get("summary_max_new_tokens", 96))
        self.temperature = float(config.inference.temperature)
        self.top_p = float(config.inference.top_p)
        self.max_new_tokens = int(config.inference.max_new_tokens)
        self._clip: Optional[_CLIPScorer] = None

    def _get_clip(self) -> Optional[_CLIPScorer]:
        if self._clip is None:
            self._clip = _CLIPScorer()
        return self._clip if self._clip.enabled else None

    def _summarize(self, video_path: str) -> str:
        # Coarse pass: 8 uniform frames -> one-sentence summary anchor.
        frames, _ = self.reader.load_and_sample(video_path, 8)
        imgs = frames_to_pil(frames)
        text, _ = self.vlm.generate(
            images=imgs,
            text=VIDEO_SUMMARY_ANCHOR_PROMPT,
            max_new_tokens=self.summary_max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return text.strip()

    def answer(
        self,
        question: str,
        video_path: str,
        options: List[str],
        subtitle: Optional[str] = None,
        benchmark: str = "video-mme-v2",
    ) -> Dict[str, Any]:
        # 1. Candidate pool + summary anchor.
        cand_frames, cand_ts = self.reader.load_and_sample(video_path, self.candidate_frames)
        cand_imgs = frames_to_pil(cand_frames)
        summary = self._summarize(video_path)

        clip = self._get_clip()
        if clip is None:
            # Fallback: uniform subset (no surprise ranking).
            select_idxs = uniform_sample(len(cand_imgs), self.select_k)
        else:
            sims = clip.similarities(cand_imgs, summary)
            surprise = 1.0 - sims  # high = surprising
            select_idxs = torch.topk(surprise, k=min(self.select_k, len(cand_imgs))).indices.tolist()
            # Keep chronological order.
            select_idxs = sorted(select_idxs)

        chosen_imgs = [cand_imgs[i] for i in select_idxs]
        prompt = build_qa_prompt(
            benchmark=benchmark,
            question=question,
            options=options,
            subtitle=subtitle,
            include_cot=True,
        )
        answer_text, n_tok = self.vlm.generate(
            images=chosen_imgs,
            text=prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        letter = extract_answer_letter(answer_text)
        return {
            "prediction": letter or "",
            "reasoning": answer_text,
            "n_tokens": n_tok,
            "n_frames_used": len(chosen_imgs),
            "tool_calls": [
                {"type": "vap_summary", "text": summary},
                {"type": "vap_selection",
                 "n_candidates": len(cand_imgs),
                 "selected_idxs": [int(i) for i in select_idxs],
                 "prior": "clip_neg_sim" if clip is not None else "uniform_fallback"},
            ],
        }
