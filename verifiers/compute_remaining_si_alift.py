"""Deterministic SI + A-lift over the remaining anchors (workflow-infra-independent).
Uses the exact intersection-SI method triple-verified on MLVU. Key = (video_id,
question_id); pool falls back to meta.question when question_id is null.
"""
import os
import json, sys
from collections import Counter

ROOT = os.environ["VRITS_ROOT"]
SJ = f"{ROOT}/analysis/standalone_juries"
E127 = f"{ROOT}/experiments/127_bon8_zeroshot_27b_tempcompass/eval/tempcompass/step_000000"
E128 = f"{ROOT}/experiments/128_bon8_zeroshot_27b_vrbench_full/eval/vrbench/step_000000"
E129 = f"{ROOT}/experiments/129_bon8_zeroshot_27b_mlvu_test_full/eval/mlvu_test/step_000000"
E111 = f"{ROOT}/experiments/111_bon8_verifier_zeroshot_27b_egoschema/eval/egoschema/step_000000"
E133 = f"{ROOT}/experiments/133_swap_muse_bon8_tempcompass_1000/eval/tempcompass/step_000000"
VE149 = f"{ROOT}/experiments/149_vholmes_bon8_zeroshot_27b/eval/video-holmes/step_000000"
VE151 = f"{ROOT}/experiments/151_vholmes_muse_bon8/eval/video-holmes/step_000000"


def _k(vid, qid, q):
    return (vid, qid if qid not in (None, "") else (q or ""))


def load_picks(path):
    d = {}
    try:
        for line in open(path):
            try:
                r = json.loads(line)
            except Exception:
                continue
            d[(r.get("video_id"), r.get("question_id"))] = r
    except FileNotFoundError:
        return None
    return d


def load_pool(path):
    d = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if not r.get("trajectories"):
            continue
        k = _k(r.get("video_id"), r.get("question_id"), (r.get("meta") or {}).get("question"))
        d[k] = r
    return d


def si_cell(jury_p, standalone_p, pool):
    J = load_picks(jury_p); S = load_picks(standalone_p)
    if J is None or S is None:
        return {"err": f"missing file J={J is not None} S={S is not None}"}
    inter = [k for k in J if k in S and str(J[k].get("pick", "")).strip() and str(S[k].get("pick", "")).strip()]
    n = len(inter)
    if not n:
        return {"err": "empty intersection"}
    ja = sum(J[k]["pick"].strip().upper() == str(J[k].get("gold", "")).strip().upper() for k in inter) / n
    sa = sum(S[k]["pick"].strip().upper() == str(S[k].get("gold", "")).strip().upper() for k in inter) / n
    # majority from pool (fallback key)
    poolkey = {_k(J[k].get("video_id"), J[k].get("question_id"), None): k for k in inter}
    cm = miss = 0
    for k in inter:
        pk = _k(J[k].get("video_id"), J[k].get("question_id"), None)
        pr = pool.get(pk)
        if pr is None:
            # try resolved (question text already in question_id)
            pr = pool.get((J[k].get("video_id"), J[k].get("question_id")))
        if pr is None:
            miss += 1; continue
        L = [t["prediction"].strip().upper() for t in pr["trajectories"] if str(t.get("prediction", "")).strip()]
        if not L:
            miss += 1; continue
        g = str(pr.get("answer_letter", "")).strip().upper()
        cm += int(Counter(L).most_common(1)[0][0] == g)
    denom = n - miss
    maj = cm / denom if denom else float("nan")
    return {"n": n, "jury_acc": round(ja * 100, 2), "standalone_acc": round(sa * 100, 2),
            "SI_pp": round((ja - sa) * 100, 2), "majority_acc": round(maj * 100, 2),
            "beats_majority": ja > maj, "pool_miss": miss}


def alift_cell(pick_p):
    J = load_picks(pick_p)
    if J is None:
        return {"err": "missing"}
    recs = [r for r in J.values() if str(r.get("pick", "")).strip()]
    n = len(recs)
    if not n:
        return {"err": "no picks"}
    pa = sum(r["pick"].strip().upper() == "A" for r in recs) / n
    ga = sum(str(r.get("gold", "")).strip().upper() == "A" for r in recs) / n
    acc = sum(r["pick"].strip().upper() == str(r.get("gold", "")).strip().upper() for r in recs) / n
    return {"n": n, "acc": round(acc * 100, 2), "pick_A": round(pa * 100, 2),
            "gold_A": round(ga * 100, 2), "A_lift_pp": round((pa - ga) * 100, 2)}


SI = {
    "VRBench/Muse": (f"{E128}/jury_holistic48_picks.jsonl", f"{E128}/jury_holistic48_standalone_picks.jsonl", E128),
    "VRBench/GPT": (f"{E128}/jury_gpt_sol_32_picks.jsonl", f"{E128}/jury_gpt_sol_32_standalone_picks.jsonl", E128),
    "VRBench/InternVL-38B": (f"{E128}/jury_local_InternVL338B_f48_picks.jsonl", f"{SJ}/standalone_InternVL338B_vrbench_f48.jsonl", E128),
    "VRBench/Phi": (f"{E128}/jury_xf_Phi35visionins_f24_picks.jsonl", f"{SJ}/standalone_Phi35visionins_vrbench_f24.jsonl", E128),
    "EgoSchema/InternVL-38B": (f"{E111}/jury_xf_InternVL338B_f48_picks.jsonl", f"{SJ}/standalone_InternVL338B_egoschema_f48.jsonl", E111),
    "EgoSchema/InternVL-8B": (f"{E111}/jury_xf_InternVL38B_f48_picks.jsonl", f"{SJ}/standalone_InternVL38B_egoschema_f48.jsonl", E111),
    "EgoSchema/Phi": (f"{E111}/jury_xf_Phi35visionins_f16_picks.jsonl", f"{SJ}/standalone_Phi35visionins_egoschema_f16.jsonl", E111),
    # --- TempCompass 2x2 + ladder: Qwen-base pool exp127 (1580) + Muse-base pool exp133 (1000) ---
    "TC-Qbase/Qwen": (f"{E127}/jury_local_Qwen3527B_f48_picks.jsonl", f"{SJ}/standalone_Qwen3527B_tempcompass_f48.jsonl", E127),
    "TC-Qbase/InternVL-38B": (f"{E127}/jury_local_InternVL338B_f48_picks.jsonl", f"{SJ}/standalone_InternVL338B_tempcompass_f48.jsonl", E127),
    "TC-Qbase/InternVL-8B": (f"{E127}/jury_local_InternVL38B_f48_picks.jsonl", f"{SJ}/standalone_InternVL38B_tempcompass_f48.jsonl", E127),
    "TC-Qbase/Phi": (f"{E127}/jury_xf_Phi35visionins_f24_picks.jsonl", f"{SJ}/standalone_Phi35visionins_tempcompass_f24.jsonl", E127),
    "TC-Qbase/Muse": (f"{E127}/jury_holistic48_picks.jsonl", f"{E127}/jury_holistic48_standalone_picks.jsonl", E127),
    "TC-Qbase/GPT": (f"{E127}/jury_gpt_sol_32_picks.jsonl", f"{E127}/jury_gpt_sol_32_standalone_picks.jsonl", E127),
    "TC-Mbase/Qwen": (f"{E133}/jury_local_Qwen3527B_f48_picks.jsonl", f"{SJ}/standalone_Qwen3527B_tempcompass_f48.jsonl", E133),
    "TC-Mbase/InternVL-38B": (f"{E133}/jury_local_InternVL338B_f48_picks.jsonl", f"{SJ}/standalone_InternVL338B_tempcompass_f48.jsonl", E133),
    "TC-Mbase/InternVL-8B": (f"{E133}/jury_local_InternVL38B_f48_picks.jsonl", f"{SJ}/standalone_InternVL38B_tempcompass_f48.jsonl", E133),
    "TC-Mbase/Phi": (f"{E133}/jury_xf_Phi35visionins_f24_picks.jsonl", f"{SJ}/standalone_Phi35visionins_tempcompass_f24.jsonl", E133),
    "TC-Mbase/Muse": (f"{E133}/jury_holistic48_picks.jsonl", f"{E127}/jury_holistic48_standalone_picks.jsonl", E133),
    # --- Video-Holmes 2x2: Qwen-base pool exp149 (1837) + Muse-base pool exp151 (1837) ---
    "VH-Qbase/Qwen": (f"{VE149}/jury_local_Qwen3527B_f48_picks.jsonl", f"{SJ}/standalone_Qwen3527B_video-holmes_f48.jsonl", VE149),
    "VH-Qbase/Muse": (f"{VE149}/jury_holistic48_picks.jsonl", f"{VE149}/jury_holistic48_standalone_picks.jsonl", VE149),
    "VH-Mbase/Qwen": (f"{VE151}/jury_local_Qwen3527B_f48_picks.jsonl", f"{SJ}/standalone_Qwen3527B_video-holmes_f48.jsonl", VE151),
    "VH-Mbase/Muse": (f"{VE151}/jury_holistic48_picks.jsonl", f"{VE149}/jury_holistic48_standalone_picks.jsonl", VE151),
}
ALIFT = {
    "TC/Phi": f"{E127}/jury_xf_Phi35visionins_f24_picks.jsonl",
    "TC/InternVL-8B": f"{E127}/jury_local_InternVL38B_f48_picks.jsonl",
    "TC/InternVL-38B": f"{E127}/jury_local_InternVL338B_f48_picks.jsonl",
    "MLVU/Phi": f"{E129}/jury_xf_Phi35visionins_f24_picks.jsonl",
    "MLVU/Qwen-self": f"{E129}/jury_local_Qwen3527B_f48_picks.jsonl",
    "MLVU/InternVL-8B": f"{E129}/jury_local_InternVL38B_f48_picks.jsonl",
    "MLVU/InternVL-38B": f"{E129}/jury_local_InternVL338B_f48_picks.jsonl",
    "VRBench/Phi": f"{E128}/jury_xf_Phi35visionins_f24_picks.jsonl",
    "VRBench/InternVL-38B": f"{E128}/jury_local_InternVL338B_f48_picks.jsonl",
}

_poolcache = {}
def getpool(p):
    if p not in _poolcache:
        _poolcache[p] = load_pool(f"{p}/predictions.jsonl")
    return _poolcache[p]

print("=== SI (intersection) ===")
for tag, (jp, sp, pooldir) in SI.items():
    print(f"{tag:32} {si_cell(jp, sp, getpool(pooldir))}")
print("\n=== A-lift (answer-position bias) ===")
for tag, pp in ALIFT.items():
    print(f"{tag:26} {alift_cell(pp)}")
