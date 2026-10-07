"""verify.py — Phase-1 best-of-N diagnostic (post-hoc, no model calls needed for
the core metrics).

Reads one or more experiments' predictions.jsonl (records must carry a
`trajectories` list of {prediction, ...}) and reports, per run and paired:

  pass@1        mean single-sample accuracy  (E[acc of one trajectory])
  majority@N    accuracy of the majority-vote letter
  oracle pass@N fraction of questions with >=1 correct trajectory  (upper bound)
  pass@k        unbiased Chen-et-al estimator for k in {1,2,4,8,...}

Decision tree (per the plan):
  oracle ~= pass@1                -> capability/perception ceiling (tools/verifier can't help)
  oracle >> pass@1                -> good trajectories EXIST -> selection problem
  (verifier@N ~ oracle => verifier works;  ~ pass@1 => verifier-reliability bottleneck)

Usage:
  python verify.py <exp_dir_or_predictions.jsonl> [<second_run> ...]
  # e.g. python verify.py ../ ../../110_bon8_verifier_evloop_27b_vmmev2/
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _find_predictions(arg: str) -> Optional[Path]:
    p = Path(arg)
    if p.is_file():
        return p
    # search under the dir for eval/*/step_*/predictions.jsonl
    cands = sorted(p.glob("eval/**/predictions.jsonl")) or sorted(p.glob("**/predictions.jsonl"))
    return cands[0] if cands else None


def _load(pred_path: Path) -> Dict[Tuple[str, Any], Dict[str, Any]]:
    """Dedup by (video_id, question_id); keep the last record that has trajectories."""
    by: Dict[Tuple[str, Any], Dict[str, Any]] = {}
    with open(pred_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            by[(r.get("video_id"), r.get("question_id"))] = r
    return by


def _gold(r: Dict[str, Any]) -> str:
    return str(r.get("answer_letter", "")).strip().upper()


def _traj_letters(r: Dict[str, Any]) -> List[str]:
    trajs = r.get("trajectories") or []
    out = []
    for t in trajs:
        pr = str(t.get("prediction", "")).strip().upper()
        if pr:  # drop empty/unparsed
            out.append(pr)
    return out


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Chen et al. 2021): 1 - C(n-c,k)/C(n,k)."""
    if k > n:
        return float("nan")
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def analyze(pred_path: Path) -> Dict[str, Any]:
    by = _load(pred_path)
    # keep only questions with >=1 trajectory
    qs = [(k, r) for k, r in by.items() if _traj_letters(r) or (r.get("trajectories") is not None)]
    n_q = 0
    pass1_sum = 0.0
    maj_correct = 0
    oracle_correct = 0
    # pass@k accumulation
    ks = [1, 2, 4, 8, 16]
    passk_sum = {k: 0.0 for k in ks}
    passk_n = {k: 0 for k in ks}
    per_correct_counts = []  # (n, c) per question
    for (_key, r) in qs:
        gold = _gold(r)
        letters = _traj_letters(r)
        n = len(letters)
        if n == 0:
            continue
        n_q += 1
        c = sum(1 for L in letters if L == gold)
        per_correct_counts.append((n, c))
        pass1_sum += c / n
        # majority
        maj = Counter(letters).most_common(1)[0][0]
        maj_correct += int(maj == gold)
        # oracle
        oracle_correct += int(c >= 1)
        for k in ks:
            if k <= n:
                passk_sum[k] += _pass_at_k(n, c, k)
                passk_n[k] += 1
    res = {
        "path": str(pred_path),
        "n_questions": n_q,
        "pass@1": pass1_sum / max(n_q, 1),
        "majority@N": maj_correct / max(n_q, 1),
        "oracle_pass@N": oracle_correct / max(n_q, 1),
        "pass@k": {k: (passk_sum[k] / passk_n[k]) if passk_n[k] else None for k in ks},
        "N_avg": (sum(n for n, _ in per_correct_counts) / max(len(per_correct_counts), 1)),
        "_by": by,
    }
    return res


def _fmt(res: Dict[str, Any]) -> str:
    pk = res["pass@k"]
    pk_s = "  ".join(f"pass@{k}={pk[k]:.4f}" for k in [1, 2, 4, 8, 16] if pk.get(k) is not None)
    return (
        f"  n={res['n_questions']}  N_avg={res['N_avg']:.1f}\n"
        f"  pass@1(mean)={res['pass@1']:.4f}   majority@N={res['majority@N']:.4f}   "
        f"oracle_pass@N={res['oracle_pass@N']:.4f}\n"
        f"  {pk_s}\n"
        f"  >>> oracle - pass@1 = {res['oracle_pass@N'] - res['pass@1']:+.4f}   "
        f"(gap = headroom a perfect verifier could capture)"
    )


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    runs = []
    for arg in sys.argv[1:]:
        pp = _find_predictions(arg)
        if pp is None:
            print(f"!! no predictions.jsonl under {arg}")
            continue
        res = analyze(pp)
        runs.append((arg, res))
        print("=" * 72)
        print(f"RUN: {arg}\n  ({res['path']})")
        print(_fmt(res))

    if len(runs) == 2:
        (a_name, a), (b_name, b) = runs
        # paired: questions present (with trajectories) in BOTH
        common = [k for k in a["_by"] if k in b["_by"]
                  and _traj_letters(a["_by"][k]) and _traj_letters(b["_by"][k])]
        print("=" * 72)
        print(f"PAIRED (matched N={len(common)}):  A={a_name}  vs  B={b_name}")

        def _sub(by, keys, fn):
            num = 0.0
            for k in keys:
                r = by[k]
                gold = _gold(r); letters = _traj_letters(r); n = len(letters)
                c = sum(1 for L in letters if L == gold)
                num += fn(n, c, letters, gold)
            return num / max(len(keys), 1)

        for label, fn in [
            ("pass@1", lambda n, c, L, g: c / n),
            ("majority@N", lambda n, c, L, g: int(Counter(L).most_common(1)[0][0] == g)),
            ("oracle_pass@N", lambda n, c, L, g: int(c >= 1)),
        ]:
            va = _sub(a["_by"], common, fn)
            vb = _sub(b["_by"], common, fn)
            print(f"  {label:16s}  A={va:.4f}  B={vb:.4f}  (B-A={vb-va:+.4f})")
        print("  Tool-value check: does B (tool) raise oracle_pass@N above A (zeroshot)?")


if __name__ == "__main__":
    main()
