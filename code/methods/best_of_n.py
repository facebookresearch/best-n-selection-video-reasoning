"""best_of_n — sample N trajectories of a base method per question and store
them all, so a post-hoc verifier (verify.py) can compute pass@1 / majority@N /
self-verify@N / jury@N / oracle-pass@N.

This is the Phase-1 diagnostic method. It does NOT change the base method's
prompting — it only samples it N times at temperature>0 for diversity. The
stored record gains a `trajectories` list; `prediction` defaults to the majority
vote (the verifier overrides it post-hoc).

Config (via inference.*):
  base_method    : "zeroshot_cot" | "evidence_loop" | ...
  n_trajectories : N (default 8)
  temperature    : sampling temperature (default 0.8; MUST be >0 for diversity)
  n_frames, max_new_tokens, top_p : as usual

Efficiency: for a batchable base (zeroshot_cot) we use VLM.generate_batch_n
(one vLLM call batches all questions x N sequences). For non-batchable bases
(evidence_loop, which is multi-round/iterative) we run the base method N times
per question serially.
"""

from __future__ import annotations

import os

for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_v, None)

from collections import Counter
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig

from prompts import build_qa_prompt, extract_answer_letter
from video_io import VideoReader

from .vlm_backend import VLM, frames_to_pil


class Method:
    name = "best_of_n"

    def __init__(self, vlm: VLM, reader: VideoReader, config: DictConfig) -> None:
        self.vlm = vlm
        self.reader = reader
        self.cfg = config
        self.base_name = str(config.inference.base_method)
        self.n = int(config.inference.n_trajectories)
        self.temperature = float(getattr(config.inference, "temperature", 0.8) or 0.8)
        self.top_p = float(getattr(config.inference, "top_p", 1.0))
        self.max_new_tokens = int(config.inference.max_new_tokens)
        self.n_frames = int(config.inference.n_frames)
        # Instantiate the base method (used for the serial / non-batchable path).
        from methods import get_method_class
        self._base = get_method_class(self.base_name)(vlm=vlm, reader=reader, config=config)
        # Fast batched path only for zeroshot_cot on a VLM exposing generate_batch_n.
        self._batchable = self.base_name == "zeroshot_cot" and hasattr(vlm, "generate_batch_n")

    # --------------------------------------------------------------- aggregation
    def _finalize(self, trajs: List[Dict[str, Any]]) -> Dict[str, Any]:
        letters = [t["prediction"].strip().upper() for t in trajs if str(t["prediction"]).strip()]
        pred = Counter(letters).most_common(1)[0][0] if letters else ""
        return {
            "prediction": pred,                       # default = majority vote
            "reasoning": trajs[0]["reasoning"] if trajs else "",
            "n_tokens": sum(int(t["n_tokens"]) for t in trajs),
            "n_frames_used": trajs[0]["n_frames_used"] if trajs else 0,
            "tool_calls": [],
            "trajectories": trajs,                    # ALL N — for post-hoc verify.py
        }

    # ------------------------------------------------------------------- serial
    def answer(
        self,
        question: str,
        video_path: str,
        options: List[str],
        subtitle: Optional[str] = None,
        benchmark: str = "video-mme-v2",
    ) -> Dict[str, Any]:
        trajs: List[Dict[str, Any]] = []
        for _ in range(self.n):
            out = self._base.answer(
                question=question, video_path=video_path, options=options,
                subtitle=subtitle, benchmark=benchmark,
            )
            trajs.append({
                "prediction": out["prediction"], "reasoning": out["reasoning"],
                "n_tokens": out["n_tokens"], "n_frames_used": out["n_frames_used"],
                "tool_calls": out.get("tool_calls", []),
            })
        return self._finalize(trajs)

    # -------------------------------------------------------------------- batched
    def answer_batch(
        self,
        samples: List[Dict[str, Any]],
        benchmark: str = "video-mme-v2",
    ) -> List[Dict[str, Any]]:
        # Non-batchable base (e.g. evidence_loop): fall back to serial per-sample.
        if not self._batchable:
            return [
                self.answer(
                    question=s["question"], video_path=s["video_path"],
                    options=s["options"], subtitle=s.get("subtitle"), benchmark=benchmark,
                )
                for s in samples
            ]
        # Fast zeroshot BoN: decode frames per sample, one batched generate with n=N.
        batch: List = []
        n_frames_each: List[int] = []
        for s in samples:
            frames, _ = self.reader.load_and_sample(s["video_path"], self.n_frames)
            images = frames_to_pil(frames)
            n_frames_each.append(len(images))
            prompt = build_qa_prompt(
                benchmark=benchmark, question=s["question"], options=s["options"],
                subtitle=s.get("subtitle"), include_cot=True,
            )
            batch.append((images, prompt))
        results = self.vlm.generate_batch_n(
            batch=batch, n=self.n, max_new_tokens=self.max_new_tokens,
            temperature=self.temperature, top_p=self.top_p,
        )
        outs: List[Dict[str, Any]] = []
        for comps, n_frames in zip(results, n_frames_each):
            trajs = []
            for text, n_tok in comps:
                letter = extract_answer_letter(text) or ""
                trajs.append({
                    "prediction": letter, "reasoning": text, "n_tokens": n_tok,
                    "n_frames_used": n_frames, "tool_calls": [],
                })
            outs.append(self._finalize(trajs))
        return outs
