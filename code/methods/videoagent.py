"""VideoAgent (Wang et al. 2024, arXiv 2403.10517) — iterative CLIP-retrieval agent.

Loop:
  1. Uniformly sample 5 frames, caption each.
  2. VLM produces answer + confidence in {1,2,3} + retrieval query if unsure.
  3. If confidence < threshold: use CLIP to retrieve top-k frames matching the
     query, caption them, add to context, loop up to `max_rounds`.
"""

from __future__ import annotations

import os

for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_v, None)

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig
from PIL import Image

from prompts import (
    CAPTION_PROMPT,
    VIDEOAGENT_ANSWER_PROMPT,
    extract_answer_letter,
    extract_confidence,
    extract_query,
)
from video_io import VideoReader, uniform_sample

from .vlm_backend import VLM, frames_to_pil

logger = logging.getLogger(__name__)


class _CLIPRetriever:
    """Wraps openai/clip-vit-large-patch14 for text-frame similarity scoring."""

    def __init__(self, device: str = "cuda") -> None:
        from transformers import CLIPModel, CLIPProcessor

        self._device = device
        self._model_id = "openai/clip-vit-large-patch14"
        try:
            self.model = CLIPModel.from_pretrained(self._model_id).to(device).eval()
            self.processor = CLIPProcessor.from_pretrained(self._model_id)
            self.enabled = True
        except Exception as e:
            logger.warning(f"CLIP retriever unavailable ({e!r}); falling back to uniform sampling")
            self.enabled = False

    @torch.no_grad()
    def score(self, images: List[Image.Image], text: str) -> torch.Tensor:
        assert self.enabled
        inputs = self.processor(
            text=[text], images=images, return_tensors="pt", padding=True
        ).to(self._device)
        out = self.model(**inputs)
        # Cosine similarity in the shared embedding space.
        image_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        text_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        sims = (image_emb @ text_emb.T).squeeze(-1)  # [N]
        return sims.cpu()


class Method:
    name = "videoagent"

    def __init__(self, vlm: VLM, reader: VideoReader, config: DictConfig) -> None:
        self.vlm = vlm
        self.reader = reader
        self.cfg = config
        method_cfg = config.get("methods", {}).get("videoagent", {})
        self.init_frames = int(method_cfg.get("init_frames", 5))
        self.max_rounds = int(method_cfg.get("max_rounds", 5))
        self.conf_threshold = int(method_cfg.get("conf_threshold", 2))
        self.retriever_kind = method_cfg.get("retriever", "clip")
        self.top_k_retrieved = int(method_cfg.get("top_k_retrieved", 1))
        self.temperature = float(config.inference.temperature)
        self.top_p = float(config.inference.top_p)
        self.max_new_tokens = int(config.inference.max_new_tokens)
        self._retriever: Optional[_CLIPRetriever] = None

    def _get_retriever(self) -> Optional[_CLIPRetriever]:
        if self.retriever_kind != "clip":
            return None
        if self._retriever is None:
            self._retriever = _CLIPRetriever()
        return self._retriever if self._retriever.enabled else None

    def _caption(self, image: Image.Image) -> Tuple[str, int]:
        text, n_tok = self.vlm.generate(
            images=[image],
            text=CAPTION_PROMPT,
            max_new_tokens=80,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return text.strip(), n_tok

    def answer(
        self,
        question: str,
        video_path: str,
        options: List[str],
        subtitle: Optional[str] = None,
        benchmark: str = "video-mme-v2",
    ) -> Dict[str, Any]:
        all_frames, fps = self.reader.load(video_path)
        n_total = int(all_frames.shape[0])
        # 1. Init: 5 uniform frames.
        init_idxs = uniform_sample(n_total, self.init_frames)
        selected_idxs: List[int] = list(init_idxs)
        captions_by_idx: Dict[int, str] = {}
        n_tok_total = 0
        tool_calls: List[Dict[str, Any]] = []

        for idx in init_idxs:
            img = frames_to_pil(all_frames[idx:idx + 1])[0]
            cap, ntok = self._caption(img)
            captions_by_idx[int(idx)] = cap
            n_tok_total += ntok
            tool_calls.append({"type": "caption_uniform", "frame": int(idx)})

        retriever = self._get_retriever()

        final_text = ""
        for round_i in range(self.max_rounds):
            # Sort captions by timestamp for coherence.
            caption_lines = [
                f"[t={i/fps:.1f}s] {captions_by_idx[i]}"
                for i in sorted(captions_by_idx)
            ]
            prompt = VIDEOAGENT_ANSWER_PROMPT.format(
                captions="\n".join(caption_lines),
                question=question,
                options="\n".join(options),
            )
            if subtitle:
                prompt = f"Subtitles (compact):\n{subtitle[:1500]}\n\n" + prompt
            resp, n_tok = self.vlm.generate(
                images=[],
                text=prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            n_tok_total += n_tok
            final_text = resp
            conf = extract_confidence(resp) or 1
            tool_calls.append({"type": "answer_round", "round": round_i, "confidence": conf})
            if conf >= self.conf_threshold:
                break
            query = extract_query(resp) or question
            # Retrieve additional frames matching query.
            if retriever is None:
                # Fallback: add more uniform frames not yet seen.
                extra = [i for i in uniform_sample(n_total, self.init_frames * 2)
                         if i not in captions_by_idx]
                extra = extra[: self.top_k_retrieved]
            else:
                # Score a candidate pool of frames — use the ones we already
                # loaded (all_frames). To keep memory bounded, evaluate a
                # uniform 64-frame pool.
                pool_idxs = [i for i in uniform_sample(n_total, 64) if i not in captions_by_idx]
                if not pool_idxs:
                    break
                pool_imgs = frames_to_pil(all_frames[pool_idxs])
                sims = retriever.score(pool_imgs, query)
                top = torch.topk(sims, k=min(self.top_k_retrieved, len(pool_idxs))).indices.tolist()
                extra = [pool_idxs[i] for i in top]
            if not extra:
                break
            for idx in extra:
                img = frames_to_pil(all_frames[idx:idx + 1])[0]
                cap, ntok = self._caption(img)
                captions_by_idx[int(idx)] = cap
                selected_idxs.append(int(idx))
                n_tok_total += ntok
                tool_calls.append({"type": "caption_retrieved", "frame": int(idx),
                                    "query": query, "round": round_i})

        letter = extract_answer_letter(final_text)
        return {
            "prediction": letter or "",
            "reasoning": final_text,
            "n_tokens": n_tok_total,
            "n_frames_used": len(captions_by_idx),
            "tool_calls": tool_calls,
        }
