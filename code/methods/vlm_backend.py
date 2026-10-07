"""Shared VLM wrapper used by every method.

Wraps a `vllm.LLM` and exposes:

    generate(images: List[PIL.Image], text: str, max_new_tokens: int) -> (str, int)

Handles Qwen3.5 VL prompt formatting (via processor.apply_chat_template) and,
optionally, `enable_thinking=True` for Think mode. Also loads a HF processor for
image preprocessing (Qwen requires the processor even when using vLLM).
"""

from __future__ import annotations

import os

for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_v, None)

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

logger = logging.getLogger(__name__)


def frames_to_pil(frames: torch.Tensor) -> List[Image.Image]:
    """[T,H,W,C] uint8 tensor -> list of PIL RGB images."""
    if frames.dtype != torch.uint8:
        frames = frames.to(torch.uint8)
    out = []
    arr = frames.cpu().numpy()
    for i in range(arr.shape[0]):
        out.append(Image.fromarray(arr[i], mode="RGB"))
    return out


class VLM:
    """Thin vLLM + Qwen processor wrapper for image+text generation."""

    def __init__(
        self,
        hf_id: str,
        tp_size: int = 1,
        dtype: str = "bfloat16",
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.9,
        thinking_mode: bool = False,
        seed: int = 42,
        **_ignore,
    ) -> None:
        from vllm import LLM, SamplingParams  # noqa: F401  (SamplingParams reused later)

        vllm_kwargs: Dict[str, Any] = dict(
            model=hf_id,
            tensor_parallel_size=tp_size,
            dtype=dtype if dtype != "fp8" else "auto",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            seed=seed,
            # Qwen3.5 is a hybrid Mamba model: each decode sequence needs one Mamba
            # cache block. best_of_n runs n>1 samples/question (batch_size x N concurrent
            # seqs), which can exceed the default max_num_seqs and abort CUDA-graph
            # capture. Cap at 64 (proven config from later baselines). Output-neutral.
            max_num_seqs=64,
        )
        if dtype == "fp8":
            vllm_kwargs["quantization"] = "fp8"
        logger.info(f"instantiating vllm.LLM with {vllm_kwargs}")
        self.llm = LLM(**vllm_kwargs)
        self.hf_id = hf_id
        self.thinking_mode = thinking_mode
        self.seed = seed
        # Load processor for chat-template + image preprocessing.
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)

    # -------------------------------------------------------------- prompt fmt
    def _build_messages(
        self,
        images: List[Image.Image],
        text: str,
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for _ in images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]

    def _apply_chat_template(self, messages: List[Dict[str, Any]]) -> str:
        # Qwen3.5 supports `enable_thinking` in the chat template.
        try:
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.thinking_mode,
            )
        except TypeError:
            # Older processors don't accept enable_thinking — fall back.
            return self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    # --------------------------------------------------------------- generate
    def generate(
        self,
        images: List[Image.Image],
        text: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Tuple[str, int]:
        """Run a single generation and return (text, n_output_tokens)."""
        from vllm import SamplingParams

        messages = self._build_messages(images, text)
        prompt = self._apply_chat_template(messages)
        sampling = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            seed=self.seed,
        )
        multimodal_data = {"image": images} if images else None
        req = {"prompt": prompt}
        if multimodal_data is not None:
            req["multi_modal_data"] = multimodal_data
        outputs = self.llm.generate([req], sampling_params=sampling)
        out = outputs[0].outputs[0]
        return out.text, len(out.token_ids)

    def generate_batch(
        self,
        batch: List[Tuple[List[Image.Image], str]],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> List[Tuple[str, int]]:
        """Batched generation, used for frame-caption inner loops."""
        from vllm import SamplingParams

        reqs = []
        for images, text in batch:
            messages = self._build_messages(images, text)
            prompt = self._apply_chat_template(messages)
            r: Dict[str, Any] = {"prompt": prompt}
            if images:
                r["multi_modal_data"] = {"image": images}
            reqs.append(r)
        sampling = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            seed=self.seed,
        )
        outs = self.llm.generate(reqs, sampling_params=sampling)
        return [(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in outs]

    # ------------------------------------------------------- best-of-N sampling
    def generate_n(
        self,
        images: List[Image.Image],
        text: str,
        n: int = 8,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 1.0,
    ) -> List[Tuple[str, int]]:
        """Sample N diverse completions for ONE prompt in a single vLLM call
        (shared prefill + N decodes). Returns [(text, n_tokens), ...] of length n."""
        from vllm import SamplingParams

        messages = self._build_messages(images, text)
        prompt = self._apply_chat_template(messages)
        sampling = SamplingParams(
            n=n, temperature=temperature, top_p=top_p,
            max_tokens=max_new_tokens, seed=self.seed,
        )
        req: Dict[str, Any] = {"prompt": prompt}
        if images:
            req["multi_modal_data"] = {"image": images}
        outputs = self.llm.generate([req], sampling_params=sampling)
        return [(o.text, len(o.token_ids)) for o in outputs[0].outputs]

    def generate_batch_n(
        self,
        batch: List[Tuple[List[Image.Image], str]],
        n: int = 8,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 1.0,
    ) -> List[List[Tuple[str, int]]]:
        """Batched best-of-N: for each (images, text) in the batch, sample N
        completions. Returns a list (per batch item) of N (text, n_tokens) tuples.
        One vLLM call batches all items x N sequences — the efficient full-scale path."""
        from vllm import SamplingParams

        reqs = []
        for images, text in batch:
            messages = self._build_messages(images, text)
            prompt = self._apply_chat_template(messages)
            r: Dict[str, Any] = {"prompt": prompt}
            if images:
                r["multi_modal_data"] = {"image": images}
            reqs.append(r)
        sampling = SamplingParams(
            n=n, temperature=temperature, top_p=top_p,
            max_tokens=max_new_tokens, seed=self.seed,
        )
        outs = self.llm.generate(reqs, sampling_params=sampling)
        return [[(o.text, len(o.token_ids)) for o in item.outputs] for item in outs]
