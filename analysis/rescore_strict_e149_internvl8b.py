#!/usr/bin/env python3
"""Strict-selector re-score of ONE cell: VH Qbase / InternVL3-8B (exp149).

Pure stdlib. Matches canonical analyze_reframe.py norm()/candidate/majority
semantics, applies the STRICT selector policy (OOC pick = WRONG), and runs a
paired McNemar (exact two-sided binomial) of STRICT-selector vs majority@8.
"""
import os
import json, math, sys
from collections import Counter

PICKS = os.environ["VRITS_ROOT"] + "/experiments/149_vholmes_bon8_zeroshot_27b/eval/video-holmes/step_000000/jury_local_InternVL38B_f48_picks.jsonl"
POOL  = os.environ["VRITS_ROOT"] + "/experiments/149_vholmes_bon8_zeroshot_27b/eval/video-holmes/step_000000/predictions.jsonl"


def norm(x):
    if x is None:
        return None
    s = str(x).strip()
    return s[0].upper() if s else None


def majority_first_seen(preds):
    """Exclude-blank plurality; first-seen (trajectory order) tiebreak."""
    v = [p for p in preds if p]
    if not v:
        return None
    cnt = Counter(v)
    top = cnt.most_common(1)[0][1]
    for p in v:              # v is in trajectory order -> first-seen tiebreak
        if cnt[p] == top:
            return p
    return v[0]


def majority_counter(preds):
    """Counter.most_common baseline (canonical majority_of) for cross-check."""
    v = [p for p in preds if p]
    return Counter(v).most_common(1)[0][0] if v else None


def exact_two_sided_binomial(b, c):
    """Two-sided exact binomial (p=0.5) over n=b+c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(tail * 2 / (2 ** n), 1.0)


# ── Build pool registry under multiple keys ────────────────────────────────
# Register each pool record under (vid,str(top_qid)),(vid,str(meta.qid)),
# (vid, meta.question). Value = the pool record's derived fields.
pool_by_key = {}
n_pool = 0
for line in open(POOL):
    r = json.loads(line)
    n_pool += 1
    vid = r.get("video_id")
    meta = r.get("meta") or {}
    top_qid = r.get("question_id")
    meta_qid = meta.get("question_id")
    question = meta.get("question")
    traj = [norm(t.get("prediction")) for t in (r.get("trajectories") or [])]
    cands = set(t for t in traj[:8] if t)
    gold = norm(r.get("answer_letter"))
    rec = {
        "vid": vid, "cands": cands, "gold": gold,
        "traj": traj[:8],
        "maj_first": majority_first_seen(traj[:8]),
        "maj_counter": majority_counter(traj[:8]),
        "n_nonempty": sum(1 for t in traj[:8] if t),
    }
    for k in ((vid, str(top_qid)), (vid, str(meta_qid)), (vid, question)):
        if k[1] is not None:
            pool_by_key.setdefault(k, rec)

# ── Iterate picks, join, score ─────────────────────────────────────────────
n_pick_lines = 0
n_empty_pick = 0
n_no_pool = 0
n_no_nonempty_traj = 0
gold_mismatch = 0

rows = []          # restricted set rows: dict(strict, solver, ooc, maj_first, maj_counter)
ooc_examples = []
for line in open(PICKS):
    r = json.loads(line)
    n_pick_lines += 1
    vid = r.get("video_id")
    qid = r.get("question_id")
    pick = norm(r.get("pick"))
    pick_gold = norm(r.get("gold"))
    if not pick:
        n_empty_pick += 1
        continue
    rec = pool_by_key.get((vid, str(qid)))
    if rec is None:
        n_no_pool += 1
        continue
    if rec["n_nonempty"] < 1:
        n_no_nonempty_traj += 1
        continue
    gold = rec["gold"]
    if pick_gold is not None and pick_gold != gold:
        gold_mismatch += 1
    cands = rec["cands"]
    solver_c = 1 if pick == gold else 0
    strict_c = 1 if (pick in cands and pick == gold) else 0
    ooc = 1 if pick not in cands else 0
    maj_first_c = 1 if rec["maj_first"] == gold else 0
    maj_counter_c = 1 if rec["maj_counter"] == gold else 0
    if ooc and len(ooc_examples) < 6:
        ooc_examples.append((vid, qid, pick, gold, sorted(cands)))
    rows.append({
        "strict": strict_c, "solver": solver_c, "ooc": ooc,
        "maj_first": maj_first_c, "maj_counter": maj_counter_c,
    })

n = len(rows)
solver_acc = 100 * sum(x["solver"] for x in rows) / n
strict_sel_acc = 100 * sum(x["strict"] for x in rows) / n
ooc_rate = 100 * sum(x["ooc"] for x in rows) / n
maj_acc = 100 * sum(x["maj_first"] for x in rows) / n
maj_acc_counter = 100 * sum(x["maj_counter"] for x in rows) / n
delta_strict_vs_maj = strict_sel_acc - maj_acc

# McNemar strict-selector vs majority(first-seen)
b = sum(1 for x in rows if x["strict"] == 1 and x["maj_first"] == 0)
c = sum(1 for x in rows if x["strict"] == 0 and x["maj_first"] == 1)
p_exact = exact_two_sided_binomial(b, c)
beats_maj_strict = strict_sel_acc > maj_acc

# ── Diagnostics ────────────────────────────────────────────────────────────
print("=== DIAGNOSTICS ===", file=sys.stderr)
print(f"pool lines={n_pool}  pick lines={n_pick_lines}", file=sys.stderr)
print(f"empty picks={n_empty_pick}  no-pool-join={n_no_pool}  "
      f"no-nonempty-traj={n_no_nonempty_traj}  gold_mismatch={gold_mismatch}",
      file=sys.stderr)
print(f"restricted n={n}", file=sys.stderr)
print(f"solver_acc={solver_acc:.4f}  strict_sel_acc={strict_sel_acc:.4f}  "
      f"ooc_rate={ooc_rate:.4f}", file=sys.stderr)
print(f"maj_acc(first-seen)={maj_acc:.4f}  maj_acc(Counter)={maj_acc_counter:.4f}",
      file=sys.stderr)
print(f"McNemar b={b} c={c} p_exact={p_exact:.6g}", file=sys.stderr)
print("OOC examples (vid,qid,pick,gold,cands):", file=sys.stderr)
for e in ooc_examples:
    print("  ", e, file=sys.stderr)

# ── Sanity gates ───────────────────────────────────────────────────────────
assert abs(solver_acc - 57.4) <= 0.6, f"solver_acc {solver_acc:.3f} off 57.4"
assert abs(maj_acc - 58.9) <= 0.6, f"maj_acc {maj_acc:.3f} off 58.9"
print("SANITY OK", file=sys.stderr)

result = {
    "tag": "VH Qbase/InternVL3-8B",
    "method": "strict-selector (OOC=wrong); majority@8 first-seen tiebreak; "
              "join=(vid,str(qid)); pure-stdlib exact binomial McNemar",
    "n": n,
    "solver_acc": round(solver_acc, 2),
    "strict_sel_acc": round(strict_sel_acc, 2),
    "ooc_rate": round(ooc_rate, 2),
    "maj_acc": round(maj_acc, 2),
    "delta_strict_vs_maj": round(delta_strict_vs_maj, 2),
    "b": b,
    "c": c,
    "p_exact": round(p_exact, 6),
    "beats_maj_strict": beats_maj_strict,
}
print(json.dumps(result))
