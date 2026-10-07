"""Paired McNemar (selector vs majority-vote@8) for the flagship 'beats self-consistency'
cells. Uses already-collected jury picks + BoN pool trajectories. Stdlib only:
exact two-sided binomial McNemar + continuity-corrected chi2; Holm-Bonferroni across the set.

Pairing mirrors compute_remaining_si_alift.py: key=(video_id,question_id) with fallback to
meta.question when question_id is null. Majority = exclude-blank plurality
(first-seen tiebreak) over trajectories[].prediction, vs answer_letter.
"""
import os
import json, math
from collections import Counter

ROOT = os.environ["VRITS_ROOT"]
VE149 = f"{ROOT}/experiments/149_vholmes_bon8_zeroshot_27b/eval/video-holmes/step_000000"
VE151 = f"{ROOT}/experiments/151_vholmes_muse_bon8/eval/video-holmes/step_000000"


def _k(vid, qid, q):
    return (vid, qid if qid not in (None, "") else (q or ""))


def load_picks(path):
    d = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        d[(r.get("video_id"), r.get("question_id"))] = r  # last-write-wins dedup
    return d


def load_pool(path):
    # Multi-key: register each pool record under (vid,str(top_qid)), (vid,str(meta.qid)),
    # and (vid,question_text) so picks join whether they key by int/str qid (Video-Holmes)
    # or by question text (null qid).
    d = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if not r.get("trajectories"):
            continue
        vid = r.get("video_id")
        m = r.get("meta") or {}
        for qid in (r.get("question_id"), m.get("question_id")):
            if qid not in (None, ""):
                d[(vid, str(qid))] = r
        q = m.get("question")
        if q:
            d[(vid, str(q))] = r
    return d


def paired(jury_p, pooldir):
    J = load_picks(jury_p)
    P = load_pool(f"{pooldir}/predictions.jsonl")
    n = selc = majc = b = c = miss = 0
    for k, r in J.items():
        pick = str(r.get("pick", "")).strip().upper()
        if not pick:
            continue
        vid = r.get("video_id")
        q = r.get("question_id")
        pr = P.get((vid, str(q)))
        if pr is None:
            miss += 1
            continue
        L = [t["prediction"].strip().upper() for t in pr["trajectories"] if str(t.get("prediction", "")).strip()]
        if not L:
            miss += 1
            continue
        g = str(pr.get("answer_letter", "")).strip().upper() or str(r.get("gold", "")).strip().upper()
        maj = Counter(L).most_common(1)[0][0]
        sc, mc = int(pick == g), int(maj == g)
        n += 1
        selc += sc
        majc += mc
        if sc and not mc:
            b += 1
        elif mc and not sc:
            c += 1
    return {"n": n, "sel_acc": round(100 * selc / n, 2), "maj_acc": round(100 * majc / n, 2),
            "b_sel_only": b, "c_maj_only": c, "n_discordant": b + c, "pool_miss": miss}


def mcnemar_p(b, c):
    nd = b + c
    if nd == 0:
        return 1.0, 1.0
    k = min(b, c)
    p_exact = min(1.0, 2.0 * sum(math.comb(nd, i) for i in range(k + 1)) * (0.5 ** nd))
    chi2 = (abs(b - c) - 1) ** 2 / nd
    p_cc = math.erfc(math.sqrt(chi2 / 2.0)) if chi2 > 0 else 1.0
    return p_exact, p_cc


CELLS = {
    "VH GPT-5.6-Sol (62.0 vs 58.9)": (f"{VE149}/jury_gpt_sol_32_picks.jsonl", VE149),
    "VH Qwen-base x Muse (61.9 vs 59.1)": (f"{VE149}/jury_holistic48_picks.jsonl", VE149),
    "VH Muse-base x Muse (60.3 vs 59.5)": (f"{VE151}/jury_holistic48_picks.jsonl", VE151),
}

rows = []
for tag, (jp, pd) in CELLS.items():
    st = paired(jp, pd)
    pe, pcc = mcnemar_p(st["b_sel_only"], st["c_maj_only"])
    st.update({"tag": tag, "p_exact": pe, "p_cc": pcc})
    rows.append(st)

# Holm-Bonferroni across the set (on exact p), m tests
m = len(rows)
order = sorted(range(m), key=lambda i: rows[i]["p_exact"])
holm_reject = {}
prev_ok = True
for rank, idx in enumerate(order):  # rank 0-indexed
    thr = 0.05 / (m - rank)
    ok = prev_ok and (rows[idx]["p_exact"] <= thr)
    holm_reject[idx] = (ok, thr)
    prev_ok = ok

for i, st in enumerate(rows):
    ok, thr = holm_reject[i]
    print(f"{st['tag']:38} n={st['n']:5} sel={st['sel_acc']:.2f} maj={st['maj_acc']:.2f} "
          f"b={st['b_sel_only']} c={st['c_maj_only']} nd={st['n_discordant']} "
          f"p_exact={st['p_exact']:.4g} p_cc={st['p_cc']:.4g} "
          f"sig@.05={'YES' if st['p_exact']<0.05 else 'no'} "
          f"Holm(thr={thr:.4g})={'PASS' if ok else 'FAIL'} miss={st['pool_miss']}")
