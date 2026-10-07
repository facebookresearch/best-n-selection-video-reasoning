"""Metric computation for the 5 video-QA benchmarks.

Every scorer accepts a list of prediction records (as written to
`predictions_{benchmark}.jsonl` by run.py) and returns a nested dict of metrics.

Prediction record schema (each line of the .jsonl):

    {
      "video_id": str,
      "question_id": str | int | None,
      "prediction": str,          # letter A/B/C/... (or "")
      "answer_letter": str,       # ground truth letter
      "task_type": str,
      "n_tokens": int,
      "n_frames_used": int,
      "time_seconds": float,      # per-query wall time
      "meta": {...}               # from the benchmark loader
    }
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional


def _acc(records: Iterable[Dict[str, Any]]) -> float:
    total = 0
    correct = 0
    for r in records:
        total += 1
        if r.get("prediction", "").strip().upper() == r.get("answer_letter", "").strip().upper():
            correct += 1
    return correct / total if total else 0.0


def _cost_metrics(records: List[Dict[str, Any]]) -> Dict[str, float]:
    if not records:
        return {"n_frames_avg": 0.0, "tokens_avg": 0.0, "time_per_query_avg": 0.0, "n": 0}
    n = len(records)
    return {
        "n_frames_avg": sum(r.get("n_frames_used", 0) for r in records) / n,
        "tokens_avg": sum(r.get("n_tokens", 0) for r in records) / n,
        "time_per_query_avg": sum(r.get("time_seconds", 0.0) for r in records) / n,
        "n": n,
    }


# ------------------------------------------------------- Video-MME v1 scorer
def score_video_mme(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_duration: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    wo_sub: List[Dict[str, Any]] = []
    w_sub: List[Dict[str, Any]] = []
    for r in records:
        meta = r.get("meta", {})
        by_duration[str(meta.get("duration") or "unknown")].append(r)
        by_domain[str(meta.get("domain") or "unknown")].append(r)
        # Split w/wo subtitle based on whether the query received one.
        (w_sub if meta.get("used_subtitle") else wo_sub).append(r)
    return {
        "avg_acc": _acc(records),
        "avg_acc_wo_sub": _acc(wo_sub),
        "avg_acc_w_sub": _acc(w_sub),
        "per_duration": {k: _acc(v) for k, v in by_duration.items()},
        "per_domain": {k: _acc(v) for k, v in by_domain.items()},
        **_cost_metrics(records),
    }


# ---------------------------------------------------- Video-MME-v2 scorer +
# non-linear group-structured metric.
#
# Video-MME-v2 groups questions into "reasoning coherence groups" via
# `group_structure` (a list of question indices that must be answered
# consistently within a group) and "capability groups" via `group_type`.
#
# The paper describes a two-step non-linear aggregator:
#   (a) within a coherence group: FIRST-ERROR TRUNCATION — score = number
#       of consecutive correct answers starting from the group's first
#       question, divided by group length. A single early miss cascades.
#   (b) within a capability group: GEOMETRIC-MEAN-LIKE non-linear
#       aggregation — score = (prod (per_coherence_score + eps))^(1/K).
# The overall Non-Lin Score is the arithmetic mean of capability scores.
#
# If group columns are absent we log a warning and fall back to Avg Acc.


def _first_error_truncation(group_records_ordered: List[Dict[str, Any]]) -> float:
    if not group_records_ordered:
        return 0.0
    n = len(group_records_ordered)
    consec = 0
    for r in group_records_ordered:
        if r.get("prediction", "").strip().upper() == r.get("answer_letter", "").strip().upper():
            consec += 1
        else:
            break
    return consec / n


def _geo_mean_nonzero(xs: List[float], eps: float = 1e-3) -> float:
    if not xs:
        return 0.0
    logsum = sum(math.log(x + eps) for x in xs)
    return math.exp(logsum / len(xs))


def _consistency(group_records_ordered: List[Dict[str, Any]]) -> float:
    if len(group_records_ordered) <= 1:
        return 1.0
    correct = [
        r.get("prediction", "").strip().upper() == r.get("answer_letter", "").strip().upper()
        for r in group_records_ordered
    ]
    return statistics.mean([1.0 if c == correct[0] else 0.0 for c in correct])


def _coherence(group_records_ordered: List[Dict[str, Any]]) -> float:
    # A coherence group is "coherent" iff either all correct or all wrong.
    if not group_records_ordered:
        return 0.0
    correct = [
        r.get("prediction", "").strip().upper() == r.get("answer_letter", "").strip().upper()
        for r in group_records_ordered
    ]
    all_c = all(correct)
    none_c = not any(correct)
    return 1.0 if (all_c or none_c) else 0.0


def score_video_mme_v2(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Sort by (group_type, coherence_group_id_or_video, question_id) so that
    # first-error truncation sees questions in intended order.
    def _coh_key(r: Dict[str, Any]) -> str:
        m = r.get("meta") or {}
        # group_structure is a list-of-ints (a numbered structure the paper
        # uses to identify coherence groups within a video). Use it as the
        # coherence-group id, falling back to video_id if absent.
        gs = m.get("group_structure")
        if gs is None:
            return f"video={r.get('video_id')}"
        try:
            return f"video={r.get('video_id')}|struct={tuple(gs)}"
        except TypeError:
            return f"video={r.get('video_id')}|struct={gs}"

    def _q_ord(r: Dict[str, Any]) -> Any:
        qid = r.get("question_id") or (r.get("meta") or {}).get("question_id") or ""
        return str(qid)

    metrics: Dict[str, Any] = {
        "avg_acc": _acc(records),
        **_cost_metrics(records),
    }

    # Level breakdown (levels 1/2/3 in Video-MME-v2).
    by_level: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_level[f"level_{(r.get('meta') or {}).get('level', 'unknown')}"].append(r)
    for k, v in by_level.items():
        metrics[k] = _acc(v)

    # Third-head breakdown (per capability sub-category).
    by_third: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        th = (r.get("meta") or {}).get("third_head") or "unknown"
        by_third[str(th)].append(r)
    metrics["per_capability"] = {k: _acc(v) for k, v in by_third.items()}

    # Non-linear metrics require group_structure. If missing, fall back.
    has_groups = any((r.get("meta") or {}).get("group_structure") is not None for r in records)
    if not has_groups:
        metrics["non_lin_score"] = metrics["avg_acc"]
        metrics["consistency"] = metrics["avg_acc"]
        metrics["coherence"] = metrics["avg_acc"]
        metrics["_notes"] = (
            "group_structure absent from records -> Non-Lin/Consistency/Coherence "
            "fall back to plain Avg Acc"
        )
        return metrics

    # Group by capability (group_type -> coherence group -> ordered records).
    by_capability: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        cap = str((r.get("meta") or {}).get("group_type") or "unknown")
        by_capability[cap][_coh_key(r)].append(r)

    consistency_scores: List[float] = []
    coherence_scores: List[float] = []
    cap_scores: List[float] = []
    for cap, coh_groups in by_capability.items():
        per_coh_scores: List[float] = []
        for _, grp in coh_groups.items():
            grp_sorted = sorted(grp, key=_q_ord)
            per_coh_scores.append(_first_error_truncation(grp_sorted))
            consistency_scores.append(_consistency(grp_sorted))
            coherence_scores.append(_coherence(grp_sorted))
        cap_scores.append(_geo_mean_nonzero(per_coh_scores))

    metrics["non_lin_score"] = statistics.mean(cap_scores) if cap_scores else 0.0
    metrics["consistency"] = statistics.mean(consistency_scores) if consistency_scores else 0.0
    metrics["coherence"] = statistics.mean(coherence_scores) if coherence_scores else 0.0
    metrics["_notes"] = (
        "non_lin_score = mean over capabilities of "
        "(geometric mean over coherence-groups of first-error-truncation acc). "
        "consistency = fraction of same-answer-within-group; "
        "coherence = fraction of groups where all-correct or all-wrong."
    )
    return metrics


# -------------------------------------------------------------------- LVBench
def score_lvbench(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_task[str(r.get("task_type") or "unknown")].append(r)
    return {
        "acc_overall": _acc(records),
        "per_task": {k: _acc(v) for k, v in by_task.items()},
        **_cost_metrics(records),
    }


# ---------------------------------------------------------------------- MLVU
_MLVU_TASKS = [
    "1_plotQA", "2_needle", "3_ego", "4_count", "5_order",
    "6_anomaly_reco", "7_topic_reasoning", "8_sub_scene", "9_summary",
]


def score_mlvu(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_task[str(r.get("task_type") or "unknown")].append(r)
    per_task = {k: _acc(v) for k, v in by_task.items()}
    # M-Avg: average over the 9 MC tasks (per MLVU convention).
    mc_tasks = [t for t in _MLVU_TASKS if t in per_task]
    m_avg = statistics.mean([per_task[t] for t in mc_tasks]) if mc_tasks else _acc(records)
    return {
        "m_avg": m_avg,
        "acc_overall": _acc(records),
        "per_task": per_task,
        **_cost_metrics(records),
    }


# ------------------------------------------------------------------- dispatch
def score_egoschema(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """EgoSchema: simple 5-way multiple-choice accuracy."""
    n = len(records)
    if n == 0:
        return {"avg_acc": 0.0, "n": 0}
    correct = sum(
        1 for r in records
        if (r.get("prediction") or "").strip().upper() == (r.get("answer_letter") or "").strip().upper()
    )
    empty = sum(1 for r in records if not (r.get("prediction") or "").strip())
    return {"avg_acc": correct / n, "n": n, "n_correct": correct, "n_empty": empty}


def score_mc_letter(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Plain multiple-choice letter-match accuracy.

    Used for the benchmarks that have no benchmark-specific scorer here
    (MLVU-Test, TempCompass, VRBench, Video-Holmes). This is the same
    convention the reported pass@1 / majority baselines use: compare the
    `prediction` letter against `answer_letter`, one row = one question, no
    task re-weighting. Official per-benchmark harnesses may weight tasks
    differently -- this reproduces the numbers reported here, not those.
    """
    n = len(records)
    if n == 0:
        return {"avg_acc": 0.0, "n": 0}
    correct = sum(
        1 for r in records
        if (r.get("prediction") or "").strip().upper() == (r.get("answer_letter") or "").strip().upper()
    )
    empty = sum(1 for r in records if not (r.get("prediction") or "").strip())
    return {"avg_acc": correct / n, "n": n, "n_correct": correct, "n_empty": empty}


SCORERS = {
    "video-mme-v2": score_video_mme_v2,
    "video-mme": score_video_mme,
    "lvbench": score_lvbench,
    "mlvu": score_mlvu,
    "egoschema": score_egoschema,
    "mlvu_test": score_mc_letter,
    "tempcompass": score_mc_letter,
    "vrbench": score_mc_letter,
    "video-holmes": score_mc_letter,
}



def score(name: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if name not in SCORERS:
        raise ValueError(f"no scorer for {name}")
    return SCORERS[name](records)
