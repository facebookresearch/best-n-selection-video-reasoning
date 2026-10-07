"""Strict-Selector re-score of ONE cell: VH Qbase/Phi-3.5 (exp149).

Policy (canonical: analysis/2026-08-01_02-47-17/scripts/analyze_reframe.py):
  candidate set for a question = distinct non-empty trajectory preds (upper)
  among the N=8 BoN trajectories. A selector pick is credited under STRICT only
  if (pick in candidates) AND (pick == gold). OOC pick = WRONG (leakage).
  majority@8 = exclude-blank first-seen plurality of the 8 traj preds.

Pure stdlib. Exact two-sided binomial (McNemar) via math.comb.
"""
import os
import json
from collections import Counter
from math import comb

D = (os.environ["VRITS_ROOT"] + "/"
     "experiments/149_vholmes_bon8_zeroshot_27b/eval/video-holmes/step_000000")
PICKS = f"{D}/jury_xf_Phi35visionins_f24_picks.jsonl"
POOL = f"{D}/predictions.jsonl"


def norm(x):
    """First non-space char, uppercased; None if empty. Matches canonical norm()."""
    if x is None:
        return None
    s = str(x).strip()
    return s[0].upper() if s else None


# ── Register pool under multiple keys (JOIN pitfall guard) ───────────────────
pool_by_key = {}
for line in open(POOL):
    r = json.loads(line)
    vid = r.get("video_id")
    meta = r.get("meta") or {}
    top_qid = r.get("question_id")
    meta_qid = meta.get("question_id")
    meta_q = meta.get("question")
    traj = [norm(t.get("prediction")) for t in (r.get("trajectories") or [])]
    gold = norm(r.get("answer_letter"))
    rec = {"traj": traj, "gold": gold}
    for k in [
        (vid, str(top_qid)) if top_qid is not None else None,
        (vid, str(meta_qid)) if meta_qid is not None else None,
        (vid, meta_q) if meta_q is not None else None,
    ]:
        if k is not None and k not in pool_by_key:
            pool_by_key[k] = rec

# ── Score ────────────────────────────────────────────────────────────────────
solver, strict, ooc, majc = [], [], [], []
n_pick_total = 0
n_empty_pick = 0
n_nojoin = 0
n_no_nonempty_traj = 0
gold_mismatch = 0

for line in open(PICKS):
    p = json.loads(line)
    vid = p.get("video_id")
    pick = norm(p.get("pick"))
    pgold = norm(p.get("gold"))
    n_pick_total += 1
    if not pick:
        n_empty_pick += 1
        continue
    rec = pool_by_key.get((vid, str(p.get("question_id"))))
    if rec is None:
        n_nojoin += 1
        continue
    cands = set(t for t in rec["traj"] if t)  # non-empty, upper
    if not cands:
        n_no_nonempty_traj += 1
        continue
    gold = rec["gold"]
    if pgold is not None and pgold != gold:
        gold_mismatch += 1  # sanity: pick.gold should == answer_letter.upper()

    solver_c = 1 if pick == gold else 0
    strict_c = 1 if (pick in cands and pick == gold) else 0
    ooc_c = 1 if pick not in cands else 0

    # majority@8: first-seen plurality among non-empty traj preds
    v = [t for t in rec["traj"] if t]
    cnt = Counter(v)
    top = cnt.most_common(1)[0][1]
    maj = next(t for t in v if cnt[t] == top)
    maj_c = 1 if maj == gold else 0

    solver.append(solver_c)
    strict.append(strict_c)
    ooc.append(ooc_c)
    majc.append(maj_c)

n = len(solver)
solver_acc = 100.0 * sum(solver) / n
strict_sel_acc = 100.0 * sum(strict) / n
ooc_rate = 100.0 * sum(ooc) / n
maj_acc = 100.0 * sum(majc) / n
delta = strict_sel_acc - maj_acc

# McNemar STRICT vs majority
b = sum(1 for s, m in zip(strict, majc) if s == 1 and m == 0)
c = sum(1 for s, m in zip(strict, majc) if s == 0 and m == 1)
nd = b + c
if nd == 0:
    p_exact = 1.0
else:
    k = min(b, c)
    tail = sum(comb(nd, i) for i in range(k + 1))
    p_exact = min(1.0, tail * 2 / (2 ** nd))

beats = strict_sel_acc > maj_acc

print("diag: picks_total=%d empty_pick=%d nojoin=%d no_nonempty_traj=%d gold_mismatch=%d"
      % (n_pick_total, n_empty_pick, n_nojoin, n_no_nonempty_traj, gold_mismatch))
print("n=%d" % n)
print("solver_acc=%.4f  (sanity ~51.8, |d|=%.3f)" % (solver_acc, abs(solver_acc - 51.8)))
print("strict_sel_acc=%.4f" % strict_sel_acc)
print("ooc_rate=%.4f" % ooc_rate)
print("maj_acc=%.4f  (sanity ~58.9, |d|=%.3f)" % (maj_acc, abs(maj_acc - 58.9)))
print("delta_strict_vs_maj=%.4f" % delta)
print("mcnemar b=%d c=%d p_exact=%.6g" % (b, c, p_exact))
print("beats_maj_strict=%s" % beats)

import json as _j
print("JSON_OUT " + _j.dumps({
    "tag": "VH Qbase/Phi-3.5", "n": n,
    "solver_acc": round(solver_acc, 4),
    "strict_sel_acc": round(strict_sel_acc, 4),
    "ooc_rate": round(ooc_rate, 4),
    "maj_acc": round(maj_acc, 4),
    "delta_strict_vs_maj": round(delta, 4),
    "b": b, "c": c, "p_exact": p_exact,
    "beats_maj_strict": beats,
    "method": "strict-selector per canonical analyze_reframe; join (vid,str(qid)); stdlib exact binomial McNemar",
}))
