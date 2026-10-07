"""TAMA — Tool-Augmented Multimodal Agent (2026-style, training-free ReAct).

Reference: TAMA (arXiv:2510.00161) + VTimeCoT (arXiv:2510.14672).

Key differences from LLoVi / VideoAgent (2023-2024):
  - No external LLM planner and no captioning stage: the VLM itself does
    interleaved CoT + tool calls in a single ReAct loop.
  - Tools return MEDIA (images/clips), not text captions -- the VLM sees the
    raw pixels of intermediate observations directly.
  - Atomic tool set (get_frame / get_clip / crop / zoom_in / search_similar)
    modeled after TAMA's Table 1.
  - Prompt-only; no fine-tuning; deterministic execution.

Loop (bounded to `max_rounds`):
  Round 0:  sample `init_frames` uniform frames + question + tool spec.
  Round r:  parse <tool .../> tags in the last assistant turn, execute each,
            append the returned images as new "user" observations, re-prompt.
  Stop:     first turn that emits <answer>X</answer>, or `max_rounds` reached.
"""

from __future__ import annotations

import os

for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_v, None)

import ast
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig
from PIL import Image

from prompts import extract_answer_letter
from video_io import VideoReader, uniform_sample

from .vlm_backend import VLM, frames_to_pil
from .videoagent import _CLIPRetriever

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- prompts
TAMA_SYSTEM_PROMPT = """You are answering a multiple-choice question about a video.

Question: {question}
Options:
{options}

You have been given {n_init} uniformly-sampled frames from the video (shown above with \
their timestamps).

You may invoke visual tools to inspect the video more carefully. Available tools \
(emit them EXACTLY in this XML-tag format, one per line):

- <tool name="get_frame" args={{"t": <seconds:float>}}/>
    Get a single frame at the given timestamp.
- <tool name="get_clip" args={{"t_start": <seconds>, "t_end": <seconds>}}/>
    Get 4 frames uniformly sampled from a time range.
- <tool name="crop" args={{"frame_idx": <int>, "bbox": [<x1>,<y1>,<x2>,<y2>]}}/>
    Crop a region of a previously-shown frame. Coords are normalized in [0,1].
- <tool name="zoom_in" args={{"frame_idx": <int>, "region_of_interest": "<what>"}}/>
    Zoom 2x on the center of a previously-shown frame.
- <tool name="search_similar" args={{"query": "<description>"}}/>
    Retrieve the top-{top_k} frames most similar to the description (CLIP retrieval).

Rules:
  1. Interleave brief reasoning ("Thought: ...") with tool calls.
  2. `frame_idx` refers to the 0-based index of a frame previously shown to you.
  3. After you have enough evidence, emit your final answer as:  <answer>X</answer>
     where X is a single letter A-H.
  4. You have at most {max_rounds} rounds of tool use. Be efficient.
{subtitle_block}
Begin your reasoning now."""


# ------------------------------------------------------------------ regex
# Match <tool name="..." args={...}/>  with either single or double quotes.
_TOOL_RE = re.compile(
    r"<tool\s+name\s*=\s*[\"']([^\"']+)[\"']\s+args\s*=\s*(\{.*?\})\s*/?\s*>",
    re.DOTALL,
)
_ANSWER_TAG_RE = re.compile(r"<answer>\s*([A-Ha-h])\s*</answer>", re.IGNORECASE)


def _parse_tool_args(raw: str) -> Optional[Dict[str, Any]]:
    """Best-effort parse of a tool args dict -- JSON first, then Python literal."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        return ast.literal_eval(raw)
    except Exception:
        return None


def _parse_tool_calls(text: str) -> List[Tuple[str, Dict[str, Any], str]]:
    """Return list of (tool_name, args_dict, raw_match) parsed from a VLM turn."""
    calls: List[Tuple[str, Dict[str, Any], str]] = []
    for m in _TOOL_RE.finditer(text):
        name = m.group(1).strip()
        args = _parse_tool_args(m.group(2))
        if args is None:
            calls.append((name, {}, m.group(0)))
        else:
            calls.append((name, args, m.group(0)))
    return calls


# ================================================================== Method
class Method:
    """TAMA-lite: training-free tool-augmented ReAct over video."""

    name = "tama"

    def __init__(self, vlm: VLM, reader: VideoReader, config: DictConfig) -> None:
        self.vlm = vlm
        self.reader = reader
        self.cfg = config
        m = config.get("methods", {}).get("tama", {})
        self.max_rounds = int(m.get("max_rounds", 4))
        self.init_frames = int(m.get("init_frames", 8))
        self.clip_pool_size = int(m.get("clip_pool_size", 32))
        self.top_k_retrieved = int(m.get("top_k_retrieved", 3))
        self.caption_max_new_tokens = int(m.get("caption_max_new_tokens", 512))
        self.temperature = float(config.inference.temperature)
        self.top_p = float(config.inference.top_p)
        self._retriever: Optional[_CLIPRetriever] = None

    # ---------------------------------------------------------------- retriever
    def _get_retriever(self) -> Optional[_CLIPRetriever]:
        if self._retriever is None:
            self._retriever = _CLIPRetriever()
        return self._retriever if self._retriever.enabled else None

    # ---------------------------------------------------------------- utilities
    def _decode_frame_at(
        self, video_path: str, t_seconds: float, n_total: int, fps: float
    ) -> Optional[Image.Image]:
        idx = max(0, min(n_total - 1, int(round(t_seconds * fps))))
        try:
            frames, _, _ = self.reader._decode_indices(video_path, [idx])
            return frames_to_pil(frames)[0]
        except Exception as e:
            logger.warning(f"decode failed at t={t_seconds}s (idx={idx}): {e!r}")
            return None

    def _decode_uniform(
        self, video_path: str, n_total: int, fps: float, k: int
    ) -> Tuple[List[Image.Image], List[float]]:
        idxs = uniform_sample(n_total, k)
        try:
            frames, _, _ = self.reader._decode_indices(video_path, idxs)
            imgs = frames_to_pil(frames)
            ts = [i / fps for i in idxs]
            return imgs, ts
        except Exception as e:
            logger.warning(f"uniform decode failed ({k} frames): {e!r}")
            return [], []

    # ---------------------------------------------------------------- tool exec
    def _tool_get_frame(
        self, args: Dict[str, Any], video_path: str, n_total: int, fps: float
    ) -> Tuple[List[Image.Image], str]:
        t = float(args["t"])
        img = self._decode_frame_at(video_path, t, n_total, fps)
        if img is None:
            return [], f"[tool call failed: could not decode frame at t={t}s]"
        return [img], f"[Tool result for get_frame(t={t:.2f}s)]"

    def _tool_get_clip(
        self, args: Dict[str, Any], video_path: str, n_total: int, fps: float
    ) -> Tuple[List[Image.Image], str]:
        t_start = float(args["t_start"])
        t_end = float(args["t_end"])
        if t_end <= t_start:
            return [], "[tool call failed: t_end must be greater than t_start]"
        n_clip_frames = 4
        ts = [t_start + (t_end - t_start) * (i + 0.5) / n_clip_frames
              for i in range(n_clip_frames)]
        idxs = [max(0, min(n_total - 1, int(round(t * fps)))) for t in ts]
        try:
            frames, _, _ = self.reader._decode_indices(video_path, idxs)
            imgs = frames_to_pil(frames)
            return imgs, f"[Tool result for get_clip(t_start={t_start:.1f}s, t_end={t_end:.1f}s), 4 frames]"
        except Exception as e:
            return [], f"[tool call failed: get_clip decode error: {e!r}]"

    def _tool_crop(
        self,
        args: Dict[str, Any],
        shown_frames: List[Image.Image],
    ) -> Tuple[List[Image.Image], str]:
        idx = int(args["frame_idx"])
        bbox = args["bbox"]
        if not (0 <= idx < len(shown_frames)):
            return [], f"[tool call failed: frame_idx {idx} out of range (0..{len(shown_frames)-1})]"
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except Exception:
            return [], "[tool call failed: bbox must be [x1,y1,x2,y2]]"
        img = shown_frames[idx]
        W, H = img.size
        # Accept either normalized [0,1] or pixel coords.
        if max(x1, y1, x2, y2) <= 1.0:
            x1p, y1p, x2p, y2p = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
        else:
            x1p, y1p, x2p, y2p = int(x1), int(y1), int(x2), int(y2)
        x1p, x2p = max(0, min(W, x1p)), max(0, min(W, x2p))
        y1p, y2p = max(0, min(H, y1p)), max(0, min(H, y2p))
        if x2p <= x1p or y2p <= y1p:
            return [], "[tool call failed: crop produced empty region]"
        cropped = img.crop((x1p, y1p, x2p, y2p))
        return [cropped], f"[Tool result for crop(frame_idx={idx}, bbox=[{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}])]"

    def _tool_zoom_in(
        self,
        args: Dict[str, Any],
        shown_frames: List[Image.Image],
    ) -> Tuple[List[Image.Image], str]:
        idx = int(args["frame_idx"])
        roi = str(args.get("region_of_interest", ""))
        if not (0 <= idx < len(shown_frames)):
            return [], f"[tool call failed: frame_idx {idx} out of range (0..{len(shown_frames)-1})]"
        img = shown_frames[idx]
        W, H = img.size
        # Approximate zoom_in: 60% center crop, upsample 2x. Logged ROI is for
        # future refinement (real grounding via OWL-ViT/GroundingDINO would slot
        # in here).
        cw, ch = int(W * 0.6), int(H * 0.6)
        x1, y1 = (W - cw) // 2, (H - ch) // 2
        cropped = img.crop((x1, y1, x1 + cw, y1 + ch))
        zoomed = cropped.resize((cw * 2, ch * 2), Image.BILINEAR)
        logger.info(f"zoom_in ROI='{roi}' on frame_idx={idx}")
        return [zoomed], f"[Tool result for zoom_in(frame_idx={idx}, region='{roi[:60]}')]"

    def _tool_search_similar(
        self, args: Dict[str, Any], video_path: str, n_total: int, fps: float
    ) -> Tuple[List[Image.Image], str, List[float]]:
        query = str(args["query"])
        retriever = self._get_retriever()
        pool_size = min(self.clip_pool_size, max(1, n_total))
        pool_idxs = uniform_sample(n_total, pool_size)
        try:
            frames, _, _ = self.reader._decode_indices(video_path, pool_idxs)
            pool_imgs = frames_to_pil(frames)
        except Exception as e:
            return [], f"[tool call failed: search_similar decode error: {e!r}]", []
        if retriever is None:
            top_imgs = pool_imgs[: self.top_k_retrieved]
            top_ts = [pool_idxs[i] / fps for i in range(len(top_imgs))]
        else:
            sims = retriever.score(pool_imgs, query)
            k = min(self.top_k_retrieved, len(pool_imgs))
            top = torch.topk(sims, k=k).indices.tolist()
            top_imgs = [pool_imgs[i] for i in top]
            top_ts = [pool_idxs[i] / fps for i in top]
        ts_str = ", ".join(f"{t:.1f}s" for t in top_ts)
        return top_imgs, f"[Tool result for search_similar(query='{query[:60]}'), top-{len(top_imgs)} frames at {ts_str}]", top_ts

    # ---------------------------------------------------------------- prompting
    def _build_prompt_text(
        self,
        question: str,
        options: List[str],
        n_init: int,
        subtitle: Optional[str],
    ) -> str:
        subtitle_block = ""
        if subtitle:
            s = subtitle.strip()
            if len(s) > 1500:
                s = s[:1500] + " ..."
            subtitle_block = f"\nSubtitles (compact):\n{s}\n"
        return TAMA_SYSTEM_PROMPT.format(
            question=question,
            options="\n".join(options) if options else "(no options)",
            n_init=n_init,
            top_k=self.top_k_retrieved,
            max_rounds=self.max_rounds,
            subtitle_block=subtitle_block,
        )

    def _build_turn_text(
        self,
        base_prompt: str,
        history: List[Dict[str, Any]],
    ) -> str:
        """Serialize the ReAct history into a single text turn.

        Each history entry is:
          {"role": "assistant"|"observation", "text": str}
        Images are supplied separately (in order) to the VLM.
        """
        parts = [base_prompt]
        for h in history:
            role = h["role"]
            if role == "assistant":
                parts.append(f"\n\n[Your previous turn]\n{h['text']}")
            elif role == "observation":
                parts.append(f"\n\n{h['text']}")
        parts.append("\n\nContinue your reasoning. If you have enough evidence, emit <answer>X</answer>.")
        return "".join(parts)

    # -------------------------------------------------------------------- main
    def answer(
        self,
        question: str,
        video_path: str,
        options: List[str],
        subtitle: Optional[str] = None,
        benchmark: str = "video-mme-v2",
    ) -> Dict[str, Any]:
        n_total, fps = self.reader._probe_n_total(video_path)
        if n_total <= 0:
            return {"prediction": "", "reasoning": "", "n_tokens": 0,
                    "n_frames_used": 0, "tool_calls": []}

        # Round 0: uniform init frames.
        init_imgs, init_ts = self._decode_uniform(video_path, n_total, fps, self.init_frames)
        shown_frames: List[Image.Image] = list(init_imgs)
        n_tok_total = 0
        tool_calls: List[Dict[str, Any]] = []
        history: List[Dict[str, Any]] = []

        base_prompt = self._build_prompt_text(
            question=question,
            options=options,
            n_init=len(init_imgs),
            subtitle=subtitle,
        )
        # Add a compact index for the init frames so the VLM can reference them.
        init_index_text = "Initial frames (frame_idx : timestamp):\n" + "\n".join(
            f"  frame_idx={i} : t={t:.1f}s" for i, t in enumerate(init_ts)
        )
        history.append({"role": "observation", "text": init_index_text})

        final_text = ""
        answered = False
        for round_i in range(self.max_rounds):
            turn_text = self._build_turn_text(base_prompt, history)
            try:
                resp, n_tok = self.vlm.generate(
                    images=shown_frames,
                    text=turn_text,
                    max_new_tokens=self.caption_max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
            except Exception as e:
                logger.warning(f"VLM generate failed at round {round_i}: {e!r}")
                break
            n_tok_total += n_tok
            final_text = resp
            history.append({"role": "assistant", "text": resp})

            # Check for final answer.
            m = _ANSWER_TAG_RE.search(resp)
            if m:
                answered = True
                tool_calls.append({"round": round_i, "type": "answer", "letter": m.group(1).upper()})
                break

            # Parse and execute tool calls.
            calls = _parse_tool_calls(resp)
            if not calls:
                # No tools and no answer: nudge once by asking for an answer.
                if round_i == self.max_rounds - 1:
                    break
                history.append({"role": "observation",
                                "text": "[no tool call detected in your last turn -- either invoke a tool or emit <answer>X</answer>]"})
                continue

            executed_any = False
            for name, args, raw in calls:
                success = True
                try:
                    if name == "get_frame":
                        imgs, obs_text = self._tool_get_frame(args, video_path, n_total, fps)
                    elif name == "get_clip":
                        imgs, obs_text = self._tool_get_clip(args, video_path, n_total, fps)
                    elif name == "crop":
                        imgs, obs_text = self._tool_crop(args, shown_frames)
                    elif name == "zoom_in":
                        imgs, obs_text = self._tool_zoom_in(args, shown_frames)
                    elif name == "search_similar":
                        imgs, obs_text, _ts = self._tool_search_similar(args, video_path, n_total, fps)
                    else:
                        imgs, obs_text = [], f"[tool call failed: unknown tool '{name}']"
                        success = False
                except KeyError as e:
                    imgs, obs_text = [], f"[tool call failed: missing arg {e}]"
                    success = False
                except Exception as e:
                    imgs, obs_text = [], f"[tool call failed: {e!r}]"
                    success = False
                if not imgs:
                    success = False

                if imgs:
                    # Register indices of the newly shown frames so the VLM can
                    # reference them via frame_idx.
                    start_idx = len(shown_frames)
                    shown_frames.extend(imgs)
                    obs_text = (
                        obs_text
                        + " (new frame_idx range: "
                        + f"{start_idx}..{start_idx + len(imgs) - 1})"
                    )
                    executed_any = True

                history.append({"role": "observation", "text": obs_text})
                tool_calls.append({
                    "round": round_i,
                    "type": "call",
                    "name": name,
                    "args": args,
                    "success": success,
                })

            if not executed_any:
                # All tool calls failed; give the model one more shot then stop.
                if round_i == self.max_rounds - 1:
                    break

        if not answered:
            # Force a final-answer turn if we exhausted rounds without one.
            history.append({"role": "observation",
                            "text": "[out of tool budget -- emit <answer>X</answer> with your best guess now]"})
            try:
                turn_text = self._build_turn_text(base_prompt, history)
                resp, n_tok = self.vlm.generate(
                    images=shown_frames,
                    text=turn_text,
                    max_new_tokens=256,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                n_tok_total += n_tok
                final_text = resp
                tool_calls.append({"round": self.max_rounds, "type": "forced_answer"})
            except Exception as e:
                logger.warning(f"forced-answer generate failed: {e!r}")

        # Extract answer letter: prefer <answer> tag; fall back to legacy regex.
        letter: Optional[str] = None
        m = _ANSWER_TAG_RE.search(final_text or "")
        if m:
            letter = m.group(1).upper()
        else:
            letter = extract_answer_letter(final_text or "")

        return {
            "prediction": letter or "",
            "reasoning": final_text,
            "n_tokens": n_tok_total,
            "n_frames_used": len(shown_frames),
            "tool_calls": tool_calls,
        }
