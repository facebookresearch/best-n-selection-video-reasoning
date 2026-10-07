#!/usr/bin/env python3
"""Video-Holmes x Qwen-27B verifier: MATCHED-ROW SI, shipped vs corrected. BOTH weak-verifier cells.

Why this exists
---------------
The two largest SI values in the whole study are Video-Holmes' weak-verifier column:
    C1  Qwen-base x Qwen-verifier  = jury 56.4  -  standalone 27.7  = +28.7
    C3  Muse-base x Qwen-verifier  = jury 58.9  -  standalone 27.7  = +31.2
BOTH arms of BOTH subtractions ran under `max_tokens=1024`. exp397/exp401 reran the two
JURY arms at 8192 with instrumented forks; exp393 has now rerun the (base-independent)
STANDALONE arm at 8192. So both SI values can finally be recomputed with every arm
defect-free, on the SAME rows.

The strong-verifier column (C2/C4) needs no correction: its standalone is Muse
`holistic48_standalone` via the Muse Spark 1.1 API (max_output_tokens floored at 16384),
and its juries are verify_jury_holistic.py:125 = 8192 -- never on the 1024 local path.

Read-only. Writes nothing except its own JSON under this directory.

Files
-----
A  shipped standalone @1024 : analysis/standalone_juries/standalone_Qwen3527B_video-holmes_f48.jsonl
B  exp393 standalone @8192  : experiments/393_.../eval/.../standalone_picks.jsonl
C  instrumented jury fork   : experiments/{397,401}_.../eval/.../jury_local_..._instrumented.jsonl
     C.pick_at1024 = CONTROL arm (reproduces the shipped 1024 behaviour)
     C.pick_fixed  = corrected arm (8192 + fixed extractor)
S  shipped jury picks       : the fork's CONTROL reference (gate). READ ONLY.

Denominator discipline
----------------------
Every number below is on the SAME matched row set (A n B n C) with the SAME full
denominator (an unparseable/empty pick counts as WRONG, never dropped). That is the only
way an SI *change* can be attributed to the defect rather than to a denominator shift.
`artifact_pp`-style mixed-denominator keys from results.json are deliberately not used.
"""
import json
import os
from math import comb

PROJ = os.environ.get(
    "PROFAI_PROJECT_DIR",
    os.environ["VRITS_ROOT"],
)

P_A = os.path.join(PROJ, "analysis/standalone_juries/standalone_Qwen3527B_video-holmes_f48.jsonl")
P_B = os.path.join(PROJ, "experiments/393_vholmes_standalone_qwen27b_budget8k/eval/"
                         "standalone_Qwen3527B_video-holmes_f48/step_0/standalone_picks.jsonl")

# The two weak-verifier cells. `shipped` is the pre-rerun pick file that the fork's
# at1024 CONTROL arm must reproduce -- it is what licenses reading the rerun as a
# counterfactual. READ ONLY (never write to these paths). NOTE the Muse-base Qwen-jury
# picks live under the POOL's dir (151), not under the jury experiment's own dir (160).
CELLS = {
    "C1": {
        "label": "Qwen-base x Qwen-verifier",
        "exp": 397,
        "fork": os.path.join(PROJ, "experiments/397_vholmes_jury_qwen27b_selfverify_budget8k/eval/"
                                   "video-holmes/step_000000/"
                                   "jury_local_Qwen3527B_f48_instrumented.jsonl"),
        "shipped": os.path.join(PROJ, "experiments/149_vholmes_bon8_zeroshot_27b/eval/video-holmes/"
                                      "step_000000/jury_local_Qwen3527B_f48_picks.jsonl"),
    },
    "C3": {
        "label": "Muse-base x Qwen-verifier",
        "exp": 401,
        "fork": os.path.join(PROJ, "experiments/401_vholmes_jury_qwen27b_over_muse_budget8k/eval/"
                                   "video-holmes/step_000000/"
                                   "jury_local_Qwen3527B_f48_instrumented.jsonl"),
        "shipped": os.path.join(PROJ, "experiments/151_vholmes_muse_bon8/eval/video-holmes/"
                                      "step_000000/jury_local_Qwen3527B_f48_picks.jsonl"),
    },
}


def norm(x):
    s = str(x or "").strip()
    return s[:1].upper() if s else None


def load(path):
    d = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            d[(r.get("video_id"), str(r.get("question_id")))] = r
    return d


def mcnemar_exact(b, c):
    """Two-sided exact binomial test on the discordant pairs (p=0.5)."""
    m = b + c
    if m == 0:
        return 1.0
    lo = min(b, c)
    tail = sum(comb(m, i) for i in range(lo + 1)) / (2.0 ** m)
    return min(1.0, 2.0 * tail)


A, B = load(P_A), load(P_B)
print(f"standalone rows: A(shipped @1024)={len(A)}  B(exp393 @8192)={len(B)}")

out = {"denominator": "full (empty/unparseable pick = wrong)", "cells": {}}

for name, spec in CELLS.items():
    C, S = load(spec["fork"]), load(spec["shipped"])
    keys = sorted(set(A) & set(B) & set(C))
    n = len(keys)

    # Gold must agree across all three files, else the join is wrong.
    mismatch = [k for k in keys
                if len({norm(A[k]["gold"]), norm(B[k]["gold"]), norm(C[k]["gold"])}) != 1]
    assert not mismatch, f"{name}: gold mismatch on {len(mismatch)} rows — join is unsound"

    def acc(get):
        """Full-denominator accuracy: an empty/unparseable pick is WRONG, not dropped."""
        return 100.0 * sum(1 for k in keys if get(k) == norm(C[k]["gold"])) / n

    std_shipped = acc(lambda k: norm(A[k]["pick"]))
    std_fixed = acc(lambda k: norm(B[k]["pick"]))
    jury_at1024 = acc(lambda k: norm(C[k]["pick_at1024"]))
    jury_fixed = acc(lambda k: norm(C[k]["pick_fixed"]))

    # CONTROL GATE: the fork's at1024 arm vs the SHIPPED pick file (S). NOTE it must NOT be
    # computed against C["pick"] -- that is the fork's own primary (8192) pick, so agreement
    # with pick_at1024 would measure the budget's effect on picks, not run-to-run
    # reproducibility. The gate's COMPLEMENT is the cross-run nondeterminism floor.
    gate_keys = [k for k in keys if k in S]
    gate = (100.0 * sum(1 for k in gate_keys if norm(C[k]["pick_at1024"]) == norm(S[k]["pick"]))
            / len(gate_keys)) if gate_keys else float("nan")

    # Truncation exposure on each arm, at the shipped 1024 cap.
    expo_jury = 100.0 * sum(1 for k in keys if C[k].get("trunc_at_shipped")) / n
    # exp393 has no at1024 arm, so exposure is inferred from token count vs the shipped cap:
    # a response that used >=1024 tokens at 8192 would have been cut off at 1024.
    expo_std = 100.0 * sum(1 for k in keys if int(B[k].get("n_tokens") or 0) >= 1024) / n
    # RESIDUAL truncation: rows the 8192 cap ALSO cut off. Any of these is still
    # defect-affected, so corrected standalone acc is a LOWER bound and corrected SI an UPPER.
    resid_std = 100.0 * sum(1 for k in keys if str(B[k].get("finish_reason")) == "length") / n
    resid_jury = 100.0 * sum(1 for k in keys if C[k].get("trunc_at_budget")) / n

    si_shipped = jury_at1024 - std_shipped
    si_corrected = jury_fixed - std_fixed

    # All four scoring policies on the SAME matched rows at the SAME full denominator, so the
    # policy ladder is a pure policy effect. STRICT abstains are scored WRONG here (not dropped)
    # -- dropping them would change the denominator and make the ladder uncomparable, which is
    # exactly the mixing that makes results.json's artifact_pp unquotable.
    policies = {
        "shipped_1024_legacy": jury_at1024 - std_shipped,
        "legacy_8192": (acc(lambda k: norm(C[k]["pick_legacy"]))
                        - acc(lambda k: norm(B[k]["pick_legacy"]))),
        "fixed_8192": si_corrected,
        "strict_8192_abstain_wrong": (acc(lambda k: norm(C[k]["pick_strict"]))
                                      - acc(lambda k: norm(B[k]["pick_strict"]))),
    }
    # Arm-level accuracies behind the strict row, so a reader can see WHY it moves.
    strict_jury = acc(lambda k: norm(C[k]["pick_strict"]))
    strict_std = acc(lambda k: norm(B[k]["pick_strict"]))

    # McNemar on the corrected SI: jury-correct vs standalone-correct over the SAME rows.
    b_only = sum(1 for k in keys if norm(C[k]["pick_fixed"]) == norm(C[k]["gold"])
                 and norm(B[k]["pick"]) != norm(C[k]["gold"]))
    c_only = sum(1 for k in keys if norm(C[k]["pick_fixed"]) != norm(C[k]["gold"])
                 and norm(B[k]["pick"]) == norm(C[k]["gold"]))
    p_si = mcnemar_exact(b_only, c_only)

    print()
    print(f"=== {name}  {spec['label']}  (exp{spec['exp']})  matched n={n} ===")
    print(f"{'arm':<34}{'acc (pp)':>10}")
    print(f"{'standalone @1024 (shipped, A)':<34}{std_shipped:>10.4f}")
    print(f"{'standalone @8192 fixed (exp393, B)':<34}{std_fixed:>10.4f}")
    print(f"{'jury @1024 CONTROL (fork)':<34}{jury_at1024:>10.4f}")
    print(f"{'jury @8192 fixed (fork)':<34}{jury_fixed:>10.4f}")
    print(f"  SI shipped   = {si_shipped:+.4f} pp")
    print(f"  SI corrected = {si_corrected:+.4f} pp   (change {si_corrected - si_shipped:+.4f})")
    print(f"  standalone-arm correction {std_fixed - std_shipped:+.4f} pp  "
          f"(exposure @1024 {expo_std:.4f}%, RESIDUAL @8192 {resid_std:.4f}%)")
    print(f"  jury-arm correction       {jury_fixed - jury_at1024:+.4f} pp  "
          f"(exposure @1024 {expo_jury:.4f}%, RESIDUAL @8192 {resid_jury:.4f}%)")
    print(f"  control gate {gate:.4f}% vs shipped picks "
          f"(complement = cross-run nondeterminism floor)")
    print(f"  McNemar on corrected SI  b={b_only} c={c_only}  p={p_si:.3e}")
    print(f"  policy ladder (same rows, same full denominator):")
    for pname, pval in policies.items():
        print(f"    SI {pname:<28}{pval:+8.4f} pp")
    print(f"    (strict arms: jury {strict_jury:.4f} / standalone {strict_std:.4f}; "
          f"abstain scored WRONG)")

    out["cells"][name] = {
        "label": spec["label"], "exp": spec["exp"], "n_matched": n,
        "standalone_shipped_1024_pp": round(std_shipped, 4),
        "standalone_fixed_8192_pp": round(std_fixed, 4),
        "jury_at1024_control_pp": round(jury_at1024, 4),
        "jury_fixed_8192_pp": round(jury_fixed, 4),
        "si_shipped_pp": round(si_shipped, 4),
        "si_corrected_pp": round(si_corrected, 4),
        "si_change_pp": round(si_corrected - si_shipped, 4),
        "standalone_arm_correction_pp": round(std_fixed - std_shipped, 4),
        "jury_arm_correction_pp": round(jury_fixed - jury_at1024, 4),
        "standalone_exposure_at1024_pct": round(expo_std, 4),
        "jury_exposure_at1024_pct": round(expo_jury, 4),
        "standalone_residual_trunc_at8192_pct": round(resid_std, 4),
        "jury_residual_trunc_at8192_pct": round(resid_jury, 4),
        "control_gate_pct": round(gate, 4),
        "si_corrected_mcnemar": {"b_jury_only": b_only, "c_standalone_only": c_only,
                                 "p_exact_two_sided": p_si},
        "si_by_policy_pp": {k: round(v, 4) for k, v in policies.items()},
        "strict_arms_pp": {"jury": round(strict_jury, 4), "standalone": round(strict_std, 4),
                           "note": "abstain scored WRONG (full denominator), not dropped"},
        "sources": {"A_shipped_standalone": P_A, "B_exp393_standalone": P_B,
                    "C_fork": spec["fork"], "S_shipped_jury_for_gate": spec["shipped"]},
    }

out["caveat"] = ("The standalone arm has no at1024 counterfactual (exp393 is a fresh run, not an "
                 "instrumented fork), so std@8192-vs-std@1024 is a CROSS-RUN comparison carrying "
                 "the ~0.16-0.32pp nondeterminism floor on top of the budget treatment. The jury "
                 "arms ARE within-fork (gates reported above). exposure_at1024 for the standalone "
                 "is INFERRED from n_tokens>=1024, not measured by a control arm.")
out["bound_direction"] = ("si_corrected is an UPPER bound on the defect-free SI: the standalone "
                          "arm is STILL truncated on standalone_residual_trunc_at8192_pct of rows "
                          "at the 8192 cap, which depresses standalone accuracy and therefore "
                          "inflates SI. Lifting the cap further can only move SI down, not up.")
out["strong_verifier_column"] = ("C2/C4 need no correction: standalone is Muse holistic48 via the "
                                 "Muse Spark 1.1 API (max_output_tokens floored 16384) and the "
                                 "juries are verify_jury_holistic.py:125 = 8192 -- never on the "
                                 "1024 local path. Clean as shipped, not merely unmeasured.")

dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matched_si.json")
with open(dst, "w") as fh:
    json.dump(out, fh, indent=2)
print(f"\nwrote {dst}")
