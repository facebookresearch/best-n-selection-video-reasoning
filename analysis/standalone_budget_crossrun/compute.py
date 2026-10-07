"""Cross-run STANDALONE budget correction: shipped(legacy@1024) -> rerun(fixed@8192).

Why this is a SEPARATE measurement from the jury forks (widget #6's `MEASURED`):
the jury reruns are instrumented FORKS -- one run emits pick_at1024 / pick_legacy /
pick_fixed on the SAME rows, so `fixed@8192 - at1024` is a within-fork contrast with
no cross-run nondeterminism, and the control gate (shipped.pick == fork.pick_at1024)
measures that nondeterminism separately.

exp393/394/395 are NOT forks. They are fresh standalone runs at max_new_tokens=8192,
so they have NO at1024 arm and NO control gate. Their correction is therefore
CROSS-RUN (shipped run vs rerun) and carries run-to-run nondeterminism inside it.
It must never be differenced against a within-fork value.

exposure is defined exactly as the jury forks define it -- the fraction of rows whose
8192-budget response is >= 1024 tokens, i.e. the rows the 1024 cap WOULD have cut.
The shipped standalone files carry only {question_id, video_id, gold, pick}, so
exposure is not computable from them; it comes from the rerun's own n_tokens.

READ ONLY. This script opens the shipped standalone files under
analysis/standalone_juries/ and each rerun's standalone_picks.jsonl and never writes
to either; its only output is results.json in this directory.
"""
import json
import os

PROJ = os.environ.get(
    "PROFAI_PROJECT_DIR",
    os.environ["VRITS_ROOT"])
HERE = os.path.dirname(os.path.abspath(__file__))

CELLS = [
    dict(exp=393, bench="Video-Holmes", n_expect=1837,
         shipped="analysis/standalone_juries/standalone_Qwen3527B_video-holmes_f48.jsonl",
         rerun="experiments/393_vholmes_standalone_qwen27b_budget8k/eval/"
               "standalone_Qwen3527B_video-holmes_f48/step_0/standalone_picks.jsonl"),
    dict(exp=394, bench="MLVU-Test", n_expect=502,
         shipped="analysis/standalone_juries/standalone_Qwen3527B_mlvu_test_f48.jsonl",
         rerun="experiments/394_mlvu_standalone_qwen27b_budget8k/eval/"
               "standalone_Qwen3527B_mlvu_test_f48/step_0/standalone_picks.jsonl"),
    dict(exp=395, bench="TempCompass", n_expect=1579,
         shipped="analysis/standalone_juries/standalone_Qwen3527B_tempcompass_f48.jsonl",
         rerun="experiments/395_tc_standalone_qwen27b_budget8k/eval/"
               "standalone_Qwen3527B_tempcompass_f48/step_0/standalone_picks.jsonl"),
]


def load(path):
    with open(os.path.join(PROJ, path)) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def key(r):
    """Row identity. `question_id` ALONE IS NOT A KEY: on TempCompass it holds the
    question TEXT, and 1580 rows carry only 867 distinct strings, so keying on it
    silently deduplicates the benchmark down to 55% of its rows (that bug produced a
    first pass reading n=867 / +8.9965pp for exp395 before this was caught). video_id
    is required to disambiguate."""
    return (r.get("video_id"), r.get("question_id"))


def join(S, R):
    """Positional join, guarded. Both files are written by the same harness in dataset
    order, so row i is the same question in both -- ASSERTED here rather than assumed,
    by requiring identical key sequences and zero gold mismatches. A positional join
    keeps every row (a key-based join would collapse TempCompass's duplicate question
    strings); the composite key is then used only to drop exact duplicate rows, which
    is what makes n match each rerun log's own denominator."""
    if len(S) != len(R):
        raise SystemExit(f"row count mismatch {len(S)} != {len(R)}")
    if [key(r) for r in S] != [key(r) for r in R]:
        raise SystemExit("key sequences differ -- positional join not licensed")
    seen, pairs = set(), []
    for s, r in zip(S, R):
        k = key(s)
        if k in seen:
            continue
        seen.add(k)
        pairs.append((s, r))
    return pairs


def norm(v):
    """Normalize a pick to a comparable letter. '' / None = abstain (never == gold)."""
    if v is None:
        return ""
    s = str(v).strip().upper()
    return s[0] if s else ""


def main():
    report = []
    for c in CELLS:
        pairs = join(load(c["shipped"]), load(c["rerun"]))
        n = len(pairs)

        # A gold mismatch between the two files would mean the runs are not scoring the
        # same questions -- the whole comparison would be void, so it is checked, not
        # assumed.
        gold_bad = sum(1 for s, r in pairs if norm(s["gold"]) != norm(r["gold"]))

        def acc(pick_of, which):
            return 100.0 * sum(1 for p in pairs
                               if pick_of(p[which]) == norm(p[which]["gold"])) / n

        def rate(pick_of, which, letter="A"):
            return 100.0 * sum(1 for p in pairs if pick_of(p[which]) == letter) / n

        p_ship = lambda r: norm(r["pick"])          # legacy extractor @1024
        p_fix = lambda r: norm(r["pick"])           # fixed extractor @8192 (rerun's primary)
        p_leg = lambda r: norm(r["pick_legacy"])
        p_str = lambda r: norm(r["pick_strict"])

        # SHIPPED = index 0, RERUN = index 1 in each joined pair.
        trunc = sum(1 for _, r in pairs if r.get("n_tokens", 0) >= 1024)
        resid = sum(1 for _, r in pairs if r.get("n_tokens", 0) >= 8192)

        a_ship, a_fix = acc(p_ship, 0), acc(p_fix, 1)
        a_leg, a_str = acc(p_leg, 1), acc(p_str, 1)
        pa_ship, pa_fix = rate(p_ship, 0), rate(p_fix, 1)
        gold_a = 100.0 * sum(1 for s, _ in pairs if norm(s["gold"]) == "A") / n
        expo = 100.0 * trunc / n

        d = dict(exp=c["exp"], bench=c["bench"], n=n, n_expect=c["n_expect"],
                 gold_disagreements=gold_bad,
                 expo_pct=round(expo, 4), resid_pct=round(100.0 * resid / n, 4),
                 trunc_rows=trunc, resid_rows=resid,
                 acc_shipped_legacy_1024=round(a_ship, 4),
                 acc_legacy_8192=round(a_leg, 4),
                 acc_fixed_8192=round(a_fix, 4),
                 acc_strict_8192_abstain_wrong=round(a_str, 4),
                 correction_pp=round(a_fix - a_ship, 4),
                 slope_pp_per_pct=round((a_fix - a_ship) / expo, 4) if expo else None,
                 pickA_shipped=round(pa_ship, 4), pickA_fixed=round(pa_fix, 4),
                 goldA=round(gold_a, 4),
                 alift_shipped=round(pa_ship - gold_a, 4),
                 alift_fixed=round(pa_fix - gold_a, 4),
                 d_pickA_pp=round(pa_fix - pa_ship, 4))
        report.append(d)

        print(f"exp{d['exp']} {d['bench']:<13} n={n} (expect {c['n_expect']}) "
              f"gold_disagree={gold_bad}")
        print(f"    exposure {d['expo_pct']:.4f}%  residual@8192 {d['resid_pct']:.4f}%")
        print(f"    acc shipped(legacy@1024) {a_ship:.4f} -> legacy@8192 {a_leg:.4f} "
              f"-> fixed@8192 {a_fix:.4f} | strict {a_str:.4f}")
        print(f"    CROSS-RUN correction {d['correction_pp']:+.4f} pp  "
              f"slope {d['slope_pp_per_pct']:.4f} pp/%")
        print(f"    pick-A {pa_ship:.4f} -> {pa_fix:.4f} (gold-A {gold_a:.4f}) "
              f"=> A-lift {d['alift_shipped']:+.4f} -> {d['alift_fixed']:+.4f}")

    with open(os.path.join(HERE, "results.json"), "w") as fh:
        json.dump({"cells": report,
                   "note": "CROSS-RUN (shipped run vs 8192 rerun): carries run-to-run "
                           "nondeterminism. Never difference against a within-fork "
                           "jury correction."}, fh, indent=2)
    print(f"\nwrote {os.path.join(HERE, 'results.json')}")


if __name__ == "__main__":
    main()
