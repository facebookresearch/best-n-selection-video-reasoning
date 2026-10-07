"""Offline unit tests for the extractors in jury_local_instrumented.py.

No GPU, no model, no data -- pure string cases. Run before launching:
    python test_extractors_jury_local.py

Three columns under test:
  legacy -- VERBATIM copy of verify_jury_local.py:_extract. This is the CONTROL.
            Its article-fallback bug is INTENTIONAL. A case where legacy differs
            from strict is a row the shipped jury run may have scored on prose
            rather than on a verdict.
  fixed  -- marker-first, then UPPERCASE-only standalone letter, clamped to the
            question's option letters.
  strict -- marker only; abstains rather than guess.

WHY THE JURY SURFACE IS WORSE THAN THE STANDALONE ONE
----------------------------------------------------
The jury prompt feeds the model the other analyses' letter conclusions
("Analysis 1 (concludes B): ..."). So when a jury response is cut mid-sentence,
the last standalone letter in the text is frequently a QUOTED CANDIDATE LETTER
rather than the verifier's own verdict. That means `fixed` -- which removes the
lowercase-article bug -- still produces a confident-looking wrong answer on the
jury surface. Only `strict` abstains. The cases below pin that down: it is the
reason the corrected jury number must be read off `strict`/marker rows, not off
`fixed`.

Also note the parent's fallback does NOT clamp to `allowed`, so it can emit a
letter that was never on offer (e.g. 'G' on a six-way question).
"""
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "jli", str(Path(__file__).parent / "jury_local_instrumented.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

SIX = "ABCDEF"
FOUR = "ABCD"
EIGHT = "ABCDEFGH"

# (text, allowed, expect_legacy, expect_fixed, expect_strict, label)
CASES = [
    # --- clean, all three agree --------------------------------------------
    ("Answer: C", SIX, "C", "C", "C", "clean terse"),
    ("Answer: (B)", SIX, "B", "B", "B", "parenthesised"),
    ("**Answer: E**", SIX, "E", "E", "E", "markdown bold"),
    ("answer-D", SIX, "D", "D", "D", "hyphen separator"),
    ("The frames show X. Therefore, Answer: F", SIX, "F", "F", "F",
     "marker after prose"),

    # --- THE ARTICLE-FALLBACK BUG, jury flavour ----------------------------
    ("Analysis 1 claims B but the frames show the subject is holding a\nsmall",
     FOUR, "A", "B", "",
     "TRUNCATED MID-SENTENCE: legacy grabs article 'a'; fixed grabs the QUOTED "
     "candidate 'B'; only strict abstains"),
    ("Analysis 3 concludes D. Looking at frame 12, I can see a", SIX, "A", "D", "",
     "trailing article -> legacy 'A'; fixed takes quoted 'D'; strict abstains"),
    ("The analyses disagree: two say C, one says A. Weighing the evidence, a",
     SIX, "A", "A", "",
     "legacy and fixed coincide on 'A' for DIFFERENT reasons (article vs quoted)"),

    # --- quoted-candidate contamination without any article ----------------
    ("Analysis 1 (concludes B) is wrong; Analysis 2 (concludes C) is right",
     SIX, "C", "C", "",
     "no marker: both guessers take the last quoted letter; strict abstains"),

    # --- the allowed-letter clamp -----------------------------------------
    ("Answer: G", SIX, "G", "", "", "impossible letter on a six-way question: "
     "legacy emits it unclamped, fixed and strict reject"),
    ("Answer: H", EIGHT, "H", "H", "H", "same letter is legal on eight-way"),

    # --- unparseable -------------------------------------------------------
    ("I cannot determine which analysis is correct from these frames.",
     SIX, "", "", "", "no letter at all -> unparseable in all three"),
    ("", SIX, "", "", "", "empty string"),
    ("Answer: AB", SIX, "A", "", "",
     "multi-letter: legacy silently takes 'A', fixed/strict record unparseable"),
]


def main():
    fails = []
    for txt, allowed, exp_l, exp_f, exp_s, label in CASES:
        got_l = m._extract_legacy(txt)
        got_f, bad = m._extract_fixed(txt, allowed)
        got_s = m._extract_strict(txt, allowed)
        ok = (got_l == exp_l and got_f == exp_f and got_s == exp_s)
        if not ok:
            fails.append((label, (exp_l, exp_f, exp_s), (got_l, got_f, got_s)))
        flag = " <- DISAGREE" if len({got_l, got_f, got_s}) > 1 else "    "
        print(f"{'ok  ' if ok else 'FAIL'} legacy={got_l!r:4} fixed={got_f!r:4} "
              f"strict={got_s!r:4} bad={bad!r:4}{flag} | {label}")
        if not ok:
            print(f"       expected legacy={exp_l!r} fixed={exp_f!r} strict={exp_s!r}")

    print("\n-- allowed-set derivation from option lists --")
    for opts, exp in [(["A. red", "B. blue", "C. green", "D. cyan", "E. tan", "F. gray"], "ABCDEF"),
                      (["(A) x", "(B) y", "(C) z", "(D) w"], "ABCD"),
                      (["A: one", "B: two", "C: three"], "ABC"),
                      (["no letters here", "also none"], "AB")]:
        got = m._allowed_for({"meta": {"options": opts}})
        ok = got == exp
        if not ok:
            fails.append((f"allowed_for {opts[0]!r}", exp, got))
        print(f"{'ok  ' if ok else 'FAIL'} {got!r:9} (expected {exp!r}) <- {opts[0]!r} ...")

    print("\n-- prompt is byte-identical to the parent's (spot check) --")
    rec = {"meta": {"question": "What happened?", "options": ["A. x", "B. y"]},
           "trajectories": [{"prediction": "A", "reasoning": "because x"},
                            {"prediction": "B", "reasoning": "because y"}]}
    p = m._prompt(rec, 48)
    checks = [
        ("48 frames sampled uniformly", "You are shown 48 frames sampled uniformly" in p),
        ("2 independent analyses", "2 independent analyses of this video" in p),
        ("Analysis 1 (concludes A)", "Analysis 1 (concludes A): because x" in p),
        ("Analysis 2 (concludes B)", "Analysis 2 (concludes B): because y" in p),
        ("options verbatim", "A. x\nB. y" in p),
        ("terminal instruction", p.endswith("Respond with ONLY the final answer as "
                                            "'Answer: X' (a single letter).")),
    ]
    for label, ok in checks:
        if not ok:
            fails.append((f"prompt: {label}", True, ok))
        print(f"{'ok  ' if ok else 'FAIL'} prompt contains {label}")

    # 500-char reasoning truncation must be preserved exactly.
    long_rec = {"meta": {"question": "q", "options": ["A. x"]},
                "trajectories": [{"prediction": "A", "reasoning": "z" * 900}]}
    lp = m._prompt(long_rec, 48)
    ok = ("z" * 500) in lp and ("z" * 501) not in lp
    if not ok:
        fails.append(("prompt: 500-char reasoning truncation", True, ok))
    print(f"{'ok  ' if ok else 'FAIL'} reasoning truncated at exactly 500 chars")

    n_dis = sum(1 for t, a, _, _, _, _ in CASES
                if m._extract_legacy(t) != m._extract_strict(t, a))
    print(f"\n{len(CASES)} extraction cases, {n_dis} where the parent (control) "
          f"extractor disagrees with strict")
    if fails:
        print(f"FAILED: {len(fails)}")
        for f in fails:
            print(f"  {f}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
