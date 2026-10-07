"""Prompt templates and answer-letter extraction for MC video QA."""

from __future__ import annotations

import re
from typing import List, Optional

# --------------------------------------------------------------------- shared
_COT_SUFFIX = (
    "Let's think step by step. "
    "After your reasoning, give your final answer as a single letter on a new "
    "line prefixed by 'Answer: '."
)


def _format_options(options: List[str]) -> str:
    return "\n".join(options) if options else "(no options provided)"


# ---------------------------------------------------------------- MC prompts
def build_qa_prompt(
    benchmark: str,
    question: str,
    options: List[str],
    subtitle: Optional[str] = None,
    include_cot: bool = True,
) -> str:
    """Build the text portion of the VLM prompt for a MC question.

    The video (as image frames) is prepended by the caller — this only returns
    the *text* messages the VLM will condition on.
    """
    header = "Watch the video and answer the multiple-choice question below.\n"
    if benchmark == "video-mme-v2":
        header = (
            "Watch the video carefully and answer the multiple-choice question. "
            "Some questions may require audio-visual reasoning across the full "
            "video; consider events chronologically.\n"
        )
    elif benchmark == "lvbench":
        header = (
            "You are watching a long (~1 hour) video. Reason across the entire "
            "video before answering the multiple-choice question.\n"
        )
    elif benchmark == "mlvu":
        header = (
            "You are watching a long-form video. Answer the multiple-choice "
            "question about the video.\n"
        )
    body = [header]
    if subtitle:
        # Truncate very long subtitle blocks to keep the prompt bounded.
        s = subtitle.strip()
        if len(s) > 4000:
            s = s[:4000] + " ..."
        body.append(f"Subtitles:\n{s}\n")
    body.append(f"Question: {question}\n")
    body.append(f"Options:\n{_format_options(options)}\n")
    if include_cot:
        body.append(_COT_SUFFIX)
    else:
        body.append("Give your final answer as a single letter prefixed by 'Answer: '.")
    return "\n".join(body)


# ----------------------------------------------------------- caption prompts
CAPTION_PROMPT = (
    "Describe what happens in this frame in one to two sentences. "
    "Focus on people, actions, objects, and text visible in the frame."
)

CLIP_CAPTION_PROMPT = (
    "Briefly describe what happens in this short video clip in 1-3 sentences, "
    "focusing on people, actions, objects, and text."
)

SUMMARY_PROMPT = (
    "Below are short captions of successive segments of a video with "
    "timestamps. First, summarize the video in a few sentences. Then, answer "
    "the multiple-choice question. Give your final answer as a single letter "
    "prefixed by 'Answer: '.\n\n"
    "Captions:\n{captions}\n\n"
    "Question: {question}\n"
    "Options:\n{options}"
)

VIDEO_SUMMARY_ANCHOR_PROMPT = (
    "In one concise sentence, describe the overall content of this video. "
    "This summary will be used to score the relevance of individual frames."
)

VIDEOAGENT_ANSWER_PROMPT = (
    "You have viewed some frames from a video with captions. Based on them, "
    "answer the multiple-choice question below. You must ALSO report your "
    "confidence on a scale of 1-3 (1=unsure, 2=fairly confident, 3=certain).\n\n"
    "Captions:\n{captions}\n\n"
    "Question: {question}\n"
    "Options:\n{options}\n\n"
    "Respond in this exact format:\n"
    "Reasoning: <one paragraph>\n"
    "Answer: <letter>\n"
    "Confidence: <1|2|3>\n"
    "If Confidence < 2, also add a line 'Query: <what visual evidence you need>'."
)


# ------------------------------------------------------------ answer parsing
_ANSWER_LINE_RE = re.compile(r"answer\s*[:\-]\s*\(?([A-Ha-h])\)?", re.IGNORECASE)
_STANDALONE_LETTER_RE = re.compile(r"\b([A-H])\b")
_FINAL_LETTER_RE = re.compile(r"\(([A-H])\)\s*$")


def extract_answer_letter(text: str, allowed: str = "ABCDEFGH") -> Optional[str]:
    """Robustly extract a single-letter answer from a free-form model response.

    Priority:
      1. explicit 'Answer: X' anywhere,
      2. trailing '(X)',
      3. the LAST standalone letter in `allowed`,
      4. None if nothing plausible.
    """
    if not text:
        return None
    m = _ANSWER_LINE_RE.search(text)
    if m and m.group(1).upper() in allowed:
        return m.group(1).upper()
    m = _FINAL_LETTER_RE.search(text.strip())
    if m and m.group(1).upper() in allowed:
        return m.group(1).upper()
    hits = _STANDALONE_LETTER_RE.findall(text)
    for L in reversed(hits):  # prefer the LAST letter
        if L.upper() in allowed:
            return L.upper()
    return None


_CONF_RE = re.compile(r"confidence\s*[:\-]\s*([1-3])", re.IGNORECASE)


def extract_confidence(text: str) -> Optional[int]:
    m = _CONF_RE.search(text or "")
    return int(m.group(1)) if m else None


_QUERY_RE = re.compile(r"query\s*[:\-]\s*(.+?)(?:\n|$)", re.IGNORECASE)


def extract_query(text: str) -> Optional[str]:
    m = _QUERY_RE.search(text or "")
    return m.group(1).strip() if m else None
