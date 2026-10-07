#!/usr/bin/env python3
"""Strict-Selector re-score of ONE VH flagship cell.

Cell: VH Qbase / GPT-5.6-Sol (FLAGSHIP)
  picks: experiments/149_vholmes_bon8_zeroshot_27b/.../jury_gpt_sol_32_picks.jsonl
  pool:  experiments/149_vholmes_bon8_zeroshot_27b/.../predictions.jsonl

Strict-Selector policy (canonical analyze_reframe.py):
  candidate set = distinct non-empty predictions among the N=8 BoN trajectories.
  A pick is credited ONLY if (pick in candidates) AND (pick == gold).
  OOC (pick not in candidates) counts as WRONG even if == gold (verifier SOLVING,
  not selecting = leakage). Majority@8 = first-seen plurality of the 8 traj preds,
  always in-candidate.
Pure stdlib; exact two-sided binomial via math.comb (matches canonical paired_mcnemar).
"""
import os
import json
from collections import Counter
from math import comb

BASE = (os.environ["VRITS_ROOT"] + "/experiments/"
        "149_vholmes_bon8_zeroshot_27b/eval/video-holmes/step_000000")
PICKS = f"{BASE}/jury_gpt_sol_32_picks.jsonl"
POOL = f"{BASE}/predictions.jsonl"


def norm(x):
    """Canonical norm: first char, upper-cased; None/empty -> None."""
    if x is None:
        return None
    s = str(x).strip()
    return s[0].upper() if s else None


def paired_mcnemar(y_a, y_b):
    """Exact two-sided binomial McNemar (canonical). b=A&~B, c=~A&B."""
    assert len(y_a) == len(y_b)
    b = sum(1 for x, y in zip(y_a, y_b) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(y_a, y_b) if x == 0 and y == 1)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 0.0
    for i in range(k + 1):
        p += comb(n, i)
    p *= 2 / (2 ** n)
    return b, c, min(p, 1.0)


def majority_tiebreak_first(preds_):
    """First-seen plurality of candidates-with-counts (Majority@8)."""
    v = [p for p in preds_ if p is not None]
    if not v:
        return None
    cnt = Counter(v)
    top = cnt.most_common(1)[0][1]
    for p in v:
        if cnt[p] == top:
            return p
    return v[0]


# ── Build pool registry under multiple keys (JOIN-pitfall robust) ────────────
# Register each pool row under (vid,str(top_qid)), (vid,str(meta.question_id)),
# (vid, meta.question). Look up picks by (vid, str(pick.question_id)).
registry = {}
for line in open(POOL):
    r = json.loads(line)
    vid = r.get("video_id")
    top_qid = r.get("question_id")
    meta = r.get("meta") or {}
    traj_preds = [norm(t.get("prediction")) for t in (r.get("trajectories") or [])]
    rec = {
        "gold": norm(r.get("answer_letter")),
        "traj_preds": traj_preds,  # keeps counts (list, not set)
        "candidates": set(p for p in traj_preds if p is not None),  # non-empty, upper
    }
    keys = []
    if vid is not None and top_qid is not None:
        keys.append((vid, str(top_qid)))
    mqid = meta.get("question_id")
    if vid is not None and mqid is not None:
        keys.append((vid, str(mqid)))
    mq = meta.get("question")
    if vid is not None and mq is not None:
        keys.append((vid, mq))
    for k in keys:
        registry.setdefault(k, rec)  # first-registered wins; keys are 1:1 here

# ── Score picks under strict policy ──────────────────────────────────────────
solver, strict, ooc, maj_c = [], [], [], []
n = 0
gold_mismatch = 0
for line in open(PICKS):
    r = json.loads(line)
    vid = r.get("video_id")
    pqid = r.get("question_id")
    pick = norm(r.get("pick"))
    pick_gold = norm(r.get("gold"))
    if not pick:  # restrict to non-empty pick
        continue
    rec = registry.get((vid, str(pqid)))
    if rec is None:
        continue
    if not rec["candidates"]:  # need >=1 non-empty trajectory pred
        continue
    gold = rec["gold"]  # answer_letter.upper()
    if pick_gold is not None and gold is not None and pick_gold != gold:
        gold_mismatch += 1
    cands = rec["candidates"]
    solver_correct = 1 if pick == gold else 0
    strict_correct = 1 if (pick in cands and pick == gold) else 0
    is_ooc = 1 if pick not in cands else 0
    maj = majority_tiebreak_first(rec["traj_preds"])
    maj_correct = 1 if maj == gold else 0
    solver.append(solver_correct)
    strict.append(strict_correct)
    ooc.append(is_ooc)
    maj_c.append(maj_correct)
    n += 1

solver_acc = 100.0 * sum(solver) / n
strict_sel_acc = 100.0 * sum(strict) / n
ooc_rate = 100.0 * sum(ooc) / n
maj_acc = 100.0 * sum(maj_c) / n
delta_strict_vs_maj = strict_sel_acc - maj_acc
b, c, p_exact = paired_mcnemar(strict, maj_c)
beats_maj_strict = strict_sel_acc > maj_acc

# ── SANITY ───────────────────────────────────────────────────────────────────
sane_solver = abs(solver_acc - 62.0) <= 0.6
sane_maj = abs(maj_acc - 58.9) <= 0.6

print(json.dumps({
    "tag": "VH Qbase/GPT-5.6-Sol (FLAGSHIP)",
    "n": n,
    "solver_acc": round(solver_acc, 4),
    "strict_sel_acc": round(strict_sel_acc, 4),
    "ooc_rate": round(ooc_rate, 4),
    "maj_acc": round(maj_acc, 4),
    "delta_strict_vs_maj": round(delta_strict_vs_maj, 4),
    "b": b, "c": c, "p_exact": p_exact,
    "beats_maj_strict": beats_maj_strict,
    "gold_mismatch_pick_vs_pool": gold_mismatch,
    "SANITY_solver_ok(|-62|<=0.6)": sane_solver,
    "SANITY_maj_ok(|-58.9|<=0.6)": sane_maj,
}, indent=2))
