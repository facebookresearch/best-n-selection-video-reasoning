"""Instrumented cross-family holistic frame-jury -- BUDGET CONTROL FORK.

Fork of verifiers/verify_jury_xfamily.py.  Purpose: measure whether the WEAK
verifier's answer-position bias ("A-lift" = pick-A rate minus gold-A rate) is a
genuine property of the under-capable verifier or an artifact of the shipped
max_tokens=1024 generation cap.

WHY THIS RUN EXISTS
  Every CAPABLE-verifier A-lift positive measured in this project flipped sign
  once the generation budget was raised from the shipped 1024 to 8192:
      TempCompass standalone   +19.1894 -> -0.6966
      Video-Holmes jury        + 2.2319 -> -2.7763
      MLVU-Test  jury          + 5.7769 -> -2.1912   (exp398, same pool as this run)
      Video-Holmes standalone  +54.5956 -> -1.4706
  The remaining positives all belong to the WEAK verifier (Phi-3.5-vision-instruct,
  +3.0 to +8.6 across five benchmarks).  Zero-GPU telemetry says those cannot be
  truncation artifacts -- Phi-3.5 emits 0.0-2.1 mean output tokens, three orders of
  magnitude under the 1024 cap -- but that is an indirect argument.  This run
  measures it directly on the largest weak-verifier positive in the census:
  MLVU-Test x Phi-3.5 @24 frames, shipped A-lift +8.5657 (exp140).

  PREDICTION: the A-lift is essentially UNCHANGED at 8192.  If instead it collapses,
  contribution C3 ("position bias is a failure mode of under-capable verifiers")
  dies and folds into C2 (the truncation/scoring-policy artifact).

WHAT IS AND IS NOT CHANGED FROM THE PARENT
  CHANGED
    1. max_tokens: 1024 -> --max-tokens (default 8192).  THE experimental variable.
    2. Environment block: the parent pops only the four proxy vars and uses
       os.environ.setdefault for HF_HOME, and never pops SSL_CERT_FILE.  That exact
       configuration killed exp397's first attempt (job 10481250) with
       FileNotFoundError on os.environ["SSL_CERT_FILE"] inside
       ModelConfig -> get_config -> hf_api().list_repo_files().  The 7-var block
       below is transplanted verbatim from exp393/394/395/397/398, whose <=1024
       control gates reproduce the shipped picks 20/20, 39/40, 166/166 -- i.e. the
       block does not perturb greedy decoding.
       VLLM_ENABLE_V1_MULTIPROCESSING is kept at the PARENT's "0" (not the "1" used
       by the jury_local fork) for maximal fidelity to the shipped xfamily run.
    3. Output path is RELATIVE (eval/<bench>/step_000000/), never pred.parent.
       The parent computes outp = pred.parent / "jury_xf_<tag>_picks.jsonl" and opens
       it in APPEND mode; pointing it at the pool would append to and corrupt the
       shipped 42,386-byte evidence file.  launch-experiment sets CWD to this
       experiment root, so output lands here and the shipped file is never touched.
    4. Instrumentation: four pick columns per row + n_tokens + finish_reason + two
       truncation flags + the raw text, plus a control gate against the shipped picks.

  NOT CHANGED (byte-identical to the parent, so the ONLY moving part is the budget)
    the prompt builder _prompt() including its 500-char per-analysis truncation,
    the _b64 resize/JPEG encoder, the legacy extractor (kept as the CONTROL column,
    article-fallback bug included), the LLM(...) kwargs, temperature=0.0, seed=42,
    the tag derivation, the record/benchmark join and its question-text fallback.

FOUR PICK COLUMNS (identical semantics to exp397/398)
  pick / pick_legacy  the parent's extractor, bug included.  BARE `pick` IS LEGACY
                      on this family -- the 393/394/395 standalone family is the
                      other way round.  Getting this backwards inverts conclusions.
  pick_at1024         legacy extractor applied to the response TRUNCATED BACK to the
                      shipped 1024-token budget.  This is the control arm: it must
                      reproduce the shipped picks.
  pick_fixed          `answer:` regex with a (?![A-Za-z]) guard, clamped to the
                      letters this question actually offers, uppercase-only fallback.
  pick_strict         marker-only; abstains rather than guessing.  On a jury the
                      prompt quotes the other analyses' letter conclusions
                      ("Analysis 1 (concludes B): ..."), so a truncated jury
                      response's last uppercase letter is frequently a QUOTED
                      CANDIDATE letter, not the verifier's own verdict.  pick_strict
                      is the only column that abstains instead of guessing.

Usage:
  python jury_xf_instrumented.py <pool_dir> [--bench mlvu_test] [--n-samples 502]
     [--frames 24] [--model microsoft/Phi-3.5-vision-instruct] [--tp 1]
     [--dtype bfloat16] [--chunk 8] [--limit N] [--max-model-len 32768]
     [--max-img-side 448] [--jpeg-q 85] [--max-tokens 8192] [--shipped <picks.jsonl>]
"""
from __future__ import annotations
import os, sys, re, json, base64, io
from collections import Counter
from pathlib import Path

# --- env block transplanted verbatim from exp393/394/395/397/398 ------------------
# The parent pops only the 4 proxy vars and never pops SSL_CERT_FILE.  With HF_HOME
# effectively unset the snapshot is absent from the default cache, vLLM goes to the
# network, and the harness-exported SSL_CERT_FILE points at a path that does not
# exist -> FileNotFoundError in hf_api().list_repo_files().  That is exactly how
# exp397 job 10481250 died.  Numerical safety: exp393/394/395 run under this block
# and their <=1024 control gates reproduce the shipped picks 20/20, 39/40, 166/166,
# so it does not perturb greedy decoding.
for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
           "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
    os.environ.pop(_v, None)
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DECORD_EOF_RETRY_MAX"] = "20480"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # PARENT's value, not the local fork's "1"

CODE = os.environ["VRITS_ROOT"] + "/experiments/109_bon8_verifier_zeroshot_27b_vmmev2/code"
CACHE = os.environ["VRITS_DATA"] + "/video_cache"
sys.path.insert(0, CODE)

SHIPPED_BUDGET = 1024          # the budget the shipped exp140 cell ran at
_ALL = "ABCDEFGH"

# Parent's regex, unchanged, used by the CONTROL columns.
_ANS_LEGACY = re.compile(r"answer\s*[:\-]\s*\(?([A-Ha-h])\)?", re.IGNORECASE)
# Same regex with the letter-boundary guard.  Transplanted verbatim from
# experiments/393/code/standalone_parsefix.py so both SI legs share ONE parsing
# convention -- do not "improve" this without also changing 393/394/395/397/398.
_ANS_FIXED = re.compile(r"answer\s*[:\-]\s*\(?([A-Ha-h])\)?(?![A-Za-z])", re.IGNORECASE)


def _extract_legacy(t):
    """The parent's extractor, byte-for-byte.  CONTROL column -- bug included,
    do NOT fix: its job is to reproduce the shipped numbers."""
    if not t:
        return ""
    m = _ANS_LEGACY.search(t)
    if m:
        return m.group(1).upper()
    for ch in reversed(re.findall(r"\b([A-Ha-h])\b", t)):
        return ch.upper()
    return ""


def _extract_fixed(t, allowed):
    """Boundary-guarded marker, clamped to the letters this question offers.
    Returns (pick, bad_letter).  bad_letter records a marker letter that was
    outside `allowed` (out-of-candidate), which is the C2 phenomenon."""
    if not t:
        return "", ""
    bad = ""
    for m in _ANS_FIXED.finditer(t):
        ch = m.group(1).upper()
        if ch in allowed:
            return ch, ""
        bad = bad or ch
    # Uppercase-only fallback.  JURY-SPECIFIC CAVEAT: the jury prompt quotes the
    # other analyses' letter conclusions, so on a truncated jury response the last
    # uppercase letter is frequently a QUOTED CANDIDATE letter rather than the
    # verifier's own verdict.  pick_strict is the only column that abstains.
    for ch in reversed(re.findall(r"\b([A-H])\b", t)):
        if ch in allowed:
            return ch, bad
        bad = bad or ch
    return "", bad


def _extract_strict(t, allowed):
    """Marker only.  Abstains instead of guessing."""
    if not t:
        return ""
    for m in _ANS_FIXED.finditer(t):
        ch = m.group(1).upper()
        if ch in allowed:
            return ch
    return ""


def _allowed_for(rec):
    """Letters this question actually offers, read off the POOLED record's meta."""
    opts = (rec.get("meta") or {}).get("options") or []
    letters = []
    for o in opts:
        m = re.match(r"\s*\(?([A-Ha-h])\s*[\.\):]", str(o))
        if m:
            letters.append(m.group(1).upper())
    if letters:
        return "".join(sorted(set(letters)))
    return _ALL[:len(opts)] if opts else _ALL


def _argval(flag, d):
    return type(d)(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else d


def _b64(img, max_side, q):
    from PIL import Image  # noqa: F401
    w, h = img.size
    s = max(w, h)
    if s > max_side:
        img = img.resize((max(1, int(w * max_side / s)), max(1, int(h * max_side / s))))
    b = io.BytesIO()
    img.convert("RGB").save(b, format="JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def _prompt(rec, n):
    q = rec.get("meta", {}).get("question", "")
    opts = rec.get("meta", {}).get("options", [])
    cand = "\n".join(
        f"Analysis {i+1} (concludes {t.get('prediction','?')}): {str(t.get('reasoning',''))[:500]}"
        for i, t in enumerate(rec["trajectories"])
    )
    return (
        f"You are shown {n} frames sampled uniformly from a video. Use the FRAMES as the "
        f"ground-truth evidence.\n\nQuestion: {q}\nOptions:\n" + "\n".join(opts) +
        f"\n\n{len(rec['trajectories'])} independent analyses of this video were produced:\n{cand}\n\n"
        "Verify the analyses against what you actually see in the frames and decide which option is "
        "correct. Respond with ONLY the final answer as 'Answer: X' (a single letter)."
    )


# ---------------------------------------------------------------------------------
# derivation helpers
# ---------------------------------------------------------------------------------
def _key(vid, qid):
    return (str(vid), str(qid))


def _load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _index(rows, label):
    """Key-index with a uniqueness assertion (J2)."""
    idx = {}
    dups = {}
    for r in rows:
        k = _key(r.get("video_id"), r.get("question_id"))
        if k in idx:
            dups[k] = dups.get(k, 1) + 1
        idx[k] = r
    if dups:
        print(f"[derive] WARNING duplicate keys in {label}: {len(dups)} "
              f"(showing 3: {list(dups.items())[:3]})", flush=True)
    return idx, dups


def _stats(idx, keys, col):
    """(full-denominator acc, non-empty acc, n_non_empty, pick-A rate on FULL denom).
    Always report BOTH conventions; never subtract across them."""
    if not keys:
        return float("nan"), float("nan"), 0, float("nan")
    hit = sum(1 for k in keys if (idx[k].get(col) or "") == (idx[k].get("gold") or ""))
    full = 100.0 * hit / len(keys)
    ne = [k for k in keys if (idx[k].get(col) or "")]
    nef = (100.0 * sum(1 for k in ne if (idx[k].get(col) or "") == (idx[k].get("gold") or "")) / len(ne)
           if ne else float("nan"))
    pa = 100.0 * sum(1 for k in keys if (idx[k].get(col) or "") == "A") / len(keys)
    return full, nef, len(ne), pa


def _derive(outp, shipped_path, order):
    """Full derivation + results.json.  All effects on a COMMON denominator."""
    R = _load_jsonl(outp)
    RI, rdup = _index(R, "rerun")
    print(f"\n[derive] rerun rows={len(R)} unique_keys={len(RI)}", flush=True)

    res = {"n_rows": len(RI)}
    ck = [k for k in order if k in RI]          # canonical key order = pool order
    if not ck:
        print("[derive] no rows to derive", flush=True)
        return

    gold_a = 100.0 * sum(1 for k in ck if (RI[k].get("gold") or "") == "A") / len(ck)

    print(f"\n=== A. MLVU-TEST WEAK-VERIFIER JURY LEG  n={len(ck)} ===", flush=True)
    cols = [("pick_at1024", "pick_at1024 (1024 CONTROL)"),
            ("pick_legacy", "pick_legacy @budget"),
            ("pick_fixed",  "pick_fixed  @budget"),
            ("pick_strict", "pick_strict @budget")]
    out = {}
    for col, label in cols:
        full, nef, nne, pa = _stats(RI, ck, col)
        out[col] = (full, nef, nne, pa)
        print(f"  {label:28s} full {full:8.4f} | non-empty {nef:8.4f} (n={nne:4d}) "
              f"| pickA {pa:8.4f}  A-lift {pa-gold_a:+8.4f}", flush=True)
    print(f"  gold-A share {gold_a:.4f}", flush=True)

    # shipped leg, joined on the same key set
    sh_full = sh_pa = float("nan")
    n_join = gate_ok = 0
    gold_dis = 0
    mism_above = mism_below = 0
    if shipped_path and Path(shipped_path).exists():
        S = _load_jsonl(shipped_path)
        SI, _ = _index(S, "shipped")
        jk = [k for k in ck if k in SI]
        n_join = len(jk)
        if jk:
            sh_hit = sum(1 for k in jk if (SI[k].get("pick") or "") == (SI[k].get("gold") or ""))
            sh_full = 100.0 * sh_hit / len(jk)
            sh_pa = 100.0 * sum(1 for k in jk if (SI[k].get("pick") or "") == "A") / len(jk)
            gold_dis = sum(1 for k in jk
                           if str(SI[k].get("gold") or "").strip().upper()
                           != str(RI[k].get("gold") or "").strip().upper())
            print(f"  shipped (1024,legacy)        full {sh_full:8.4f} | "
                  f"pickA {sh_pa:8.4f}  A-lift {sh_pa-gold_a:+8.4f}   (joined n={n_join})", flush=True)
            print(f"  gold disagreements shipped-vs-rerun: {gold_dis}", flush=True)

            print(f"\n=== B. CONTROL GATE: shipped.pick == rerun.pick_at1024 ===", flush=True)
            shown = 0
            for k in jk:
                a = str(SI[k].get("pick", "")).strip().upper()
                b = str(RI[k].get("pick_at1024", "")).strip().upper()
                if a == b:
                    gate_ok += 1
                else:
                    nt = RI[k].get("n_tokens")
                    if isinstance(nt, int) and nt > SHIPPED_BUDGET:
                        mism_above += 1
                    else:
                        mism_below += 1
                    if shown < 25:
                        print(f"  MISMATCH {k}: shipped={a!r} at1024={b!r} n_tok={nt}", flush=True)
                        shown += 1
            print(f"  {gate_ok}/{n_join} = {100.0*gate_ok/n_join:.4f}%   "
                  f"mismatches={n_join-gate_ok} "
                  f"(truncation victims n_tok>{SHIPPED_BUDGET}: {mism_above} | "
                  f"nondeterminism n_tok<={SHIPPED_BUDGET}: {mism_below})", flush=True)
    else:
        print(f"\n[derive] no shipped file at {shipped_path!r} -- gate SKIPPED", flush=True)

    print(f"\n=== C. TRUNCATION ===", flush=True)
    wt = [k for k in ck if RI[k].get("trunc_at_shipped")]
    st = [k for k in ck if RI[k].get("trunc_at_budget")]
    print(f"  would-truncate@{SHIPPED_BUDGET}  {len(wt)}/{len(ck)} = {100.0*len(wt)/len(ck):.4f}%", flush=True)
    print(f"  still-truncated@budget {len(st)}/{len(ck)} = {100.0*len(st)/len(ck):.4f}%"
          f"   => corrected accs are LOWER bounds, corrected A-lift an UPPER bound "
          f"on A-attraction", flush=True)
    ntoks = [RI[k].get("n_tokens") for k in ck if isinstance(RI[k].get("n_tokens"), int)]
    if ntoks:
        ntoks_s = sorted(ntoks)
        print(f"  n_tokens: mean {sum(ntoks)/len(ntoks):.2f}  median {ntoks_s[len(ntoks_s)//2]}  "
              f"max {ntoks_s[-1]}   (Phi-3.5 telemetry predicted 0-2 mean)", flush=True)

    # budget / extractor decomposition, legacy-vs-legacy, common denominator
    budget_pp = out["pick_legacy"][0] - sh_full if sh_full == sh_full else float("nan")
    extract_pp = out["pick_fixed"][0] - out["pick_legacy"][0]
    print(f"\n=== D. DECOMPOSITION (same extractor / same denominator) ===", flush=True)
    print(f"  BUDGET    effect (legacy@budget - shipped@{SHIPPED_BUDGET}): {budget_pp:+.4f} pp", flush=True)
    print(f"  EXTRACTOR effect (fixed - legacy, @budget):                 {extract_pp:+.4f} pp", flush=True)

    print(f"\n=== E. PREFIX BIAS TEST (budget delta, first vs second half, POOL order) ===", flush=True)
    if sh_full == sh_full and n_join:
        h = len(jk) // 2
        for label, part in (("first half", jk[:h]), ("second half", jk[h:])):
            if not part:
                continue
            a = 100.0 * sum(1 for k in part if (RI[k].get("pick_legacy") or "") == (RI[k].get("gold") or "")) / len(part)
            b = 100.0 * sum(1 for k in part if (SI[k].get("pick") or "") == (SI[k].get("gold") or "")) / len(part)
            print(f"  {label:12s} budget delta {a-b:+8.4f}  (n={len(part)})", flush=True)

    res.update({
        "gold_a_share": round(gold_a, 4),
        "acc_shipped_1024_legacy": round(sh_full, 4) if sh_full == sh_full else None,
        "acc_at1024_control": round(out["pick_at1024"][0], 4),
        "acc_legacy_budget": round(out["pick_legacy"][0], 4),
        "acc_fixed_budget": round(out["pick_fixed"][0], 4),
        "acc_fixed_budget_nonempty": round(out["pick_fixed"][1], 4),
        "n_nonempty_fixed": out["pick_fixed"][2],
        "acc_strict_budget": round(out["pick_strict"][0], 4),
        "acc_strict_budget_nonempty": round(out["pick_strict"][1], 4),
        "n_nonempty_strict": out["pick_strict"][2],
        "pick_a_shipped": round(sh_pa, 4) if sh_pa == sh_pa else None,
        "pick_a_at1024": round(out["pick_at1024"][3], 4),
        "pick_a_legacy_budget": round(out["pick_legacy"][3], 4),
        "pick_a_fixed_budget": round(out["pick_fixed"][3], 4),
        "pick_a_strict_budget": round(out["pick_strict"][3], 4),
        "a_lift_shipped": round(sh_pa - gold_a, 4) if sh_pa == sh_pa else None,
        "a_lift_at1024": round(out["pick_at1024"][3] - gold_a, 4),
        "a_lift_legacy_budget": round(out["pick_legacy"][3] - gold_a, 4),
        "a_lift_fixed_budget": round(out["pick_fixed"][3] - gold_a, 4),
        "a_lift_strict_budget": round(out["pick_strict"][3] - gold_a, 4),
        "budget_effect_pp_same_denom": round(budget_pp, 4) if budget_pp == budget_pp else None,
        "extractor_effect_pp_same_denom": round(extract_pp, 4),
        "gate_at1024_eq_shipped_pct": round(100.0 * gate_ok / n_join, 4) if n_join else None,
        "gate_mismatches": (n_join - gate_ok) if n_join else None,
        "gate_mismatch_truncation_victims": mism_above,
        "gate_mismatch_nondeterministic": mism_below,
        "would_truncate_at_1024_pct": round(100.0 * len(wt) / len(ck), 4),
        "still_truncated_at_budget_pct": round(100.0 * len(st) / len(ck), 4),
        "mean_output_tokens": round(sum(ntoks) / len(ntoks), 4) if ntoks else None,
        "gold_disagreements_vs_shipped": gold_dis,
    })
    with open("results.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[derive] wrote results.json ({len(res)} keys)", flush=True)


def main():
    exp = sys.argv[1].rstrip("/")
    bench = _argval("--bench", "video-mme-v2")
    n_samples = _argval("--n-samples", 500)
    subset = _argval("--subset", "")
    n_frames = _argval("--frames", 48)
    model = _argval("--model", "OpenGVLab/InternVL3-38B")
    tp = _argval("--tp", 1)
    dtype = _argval("--dtype", "bfloat16")
    chunk = _argval("--chunk", 8)
    limit = _argval("--limit", 10 ** 9)
    mml = _argval("--max-model-len", 32768)
    max_side = _argval("--max-img-side", 448)
    jq = _argval("--jpeg-q", 85)
    mdp = _argval("--max-dynamic-patch", 1)
    max_tokens = _argval("--max-tokens", 8192)      # THE experimental variable
    shipped = _argval("--shipped", "")

    from vllm import LLM, SamplingParams
    from video_io import VideoReader
    from methods.vlm_backend import frames_to_pil

    from benchmarks import load_benchmark

    tag = model.split("/")[-1].replace(".", "").replace("-", "")[:14] + f"_f{n_frames}"
    pred = sorted(Path(exp).glob("eval/**/predictions.jsonl"))[0]
    # SAFETY: relative outdir under THIS experiment (CWD = experiment root).  Never
    # pred.parent -- the parent's append-mode write would corrupt the shipped picks.
    outdir = Path("eval") / bench / "step_000000"
    outdir.mkdir(parents=True, exist_ok=True)
    outp = outdir / f"jury_xf_{tag}_instrumented.jsonl"
    print(f"[jury_xf_8k] pool={pred}\n[jury_xf_8k] out={outp.resolve()}\n"
          f"[jury_xf_8k] max_tokens={max_tokens} (shipped was {SHIPPED_BUDGET})", flush=True)

    recs = {}
    for line in open(pred):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("trajectories"):
            _qid = r.get("question_id"); _qid = _qid if _qid not in (None, "") else (r.get("meta") or {}).get("question")
            recs[(r.get("video_id"), _qid)] = r
    smap = {}
    _bkw = {"subset": subset} if subset else {}
    for s in load_benchmark(bench, cache_root=CACHE, n_samples=n_samples, **_bkw):
        _sqid = s.get("meta", {}).get("question_id"); _sqid = _sqid if _sqid not in (None, "") else s.get("question")
        smap[(s["video_id"], _sqid)] = s
    done = set()
    if outp.exists():
        for line in open(outp):
            try:
                r = json.loads(line); done.add(_key(r["video_id"], r["question_id"]))
            except Exception:
                pass
    order = [_key(k[0], k[1]) for k in recs]      # canonical order = predictions.jsonl order
    todo = [(k, r) for k, r in recs.items() if _key(k[0], k[1]) not in done and k in smap][:limit]
    print(f"jury_xf[{tag}] model={model}: {len(recs)} recs, {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        _derive(outp, shipped, order)
        return

    # NOTE (parent): InternVL dynamic tiling is capped via config.json rather than
    # mm_processor_kwargs -- passing max_dynamic_patch as a kwarg crashes.
    _ = mdp  # retained for CLI compatibility
    llm = LLM(model=model, tensor_parallel_size=tp, trust_remote_code=True,
              dtype=dtype, max_model_len=mml, gpu_memory_utilization=0.90,
              limit_mm_per_prompt={"image": n_frames + 2}, max_num_seqs=max(4, chunk), seed=42)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=42)
    reader = VideoReader(cache_root=CACHE, frame_extractor="decord", frame_resize=336)

    tok = None
    try:
        tok = llm.get_tokenizer()
    except Exception as e:
        print(f"[jury_xf_8k] FATAL: no tokenizer ({e}) -> cannot reconstruct pick_at1024", flush=True)
        raise

    fout = open(outp, "a")
    for c0 in range(0, len(todo), chunk):
        convs = []
        keys = []
        for k, r in todo[c0:c0 + chunk]:
            try:
                fr, _ = reader.load_and_sample(smap[k]["video_path"], n_frames)
                imgs = frames_to_pil(fr)
            except Exception:
                fout.write(json.dumps({
                    "video_id": k[0], "question_id": k[1],
                    "gold": str(r.get("answer_letter", "")).strip().upper(),
                    "raw": "", "allowed": _allowed_for(r),
                    "pick": "", "pick_legacy": "", "pick_at1024": "",
                    "pick_fixed": "", "pick_strict": "", "bad_letter": "",
                    "n_tokens": 0, "finish_reason": "decode_error",
                    "trunc_at_shipped": False, "trunc_at_budget": False,
                    "err": "decode"}) + "\n"); fout.flush(); continue
            content = [{"type": "image_url", "image_url": {"url": _b64(im, max_side, jq)}} for im in imgs]
            content.append({"type": "text", "text": _prompt(r, len(imgs))})
            convs.append([{"role": "user", "content": content}]); keys.append((k, r))
        if not convs:
            continue
        outs = llm.chat(convs, sampling_params=sp)
        for (k, r), o in zip(keys, outs):
            co = o.outputs[0]
            txt = co.text
            tids = list(co.token_ids or [])
            n_tok = len(tids)
            fr_reason = getattr(co, "finish_reason", "") or ""
            # reconstruct exactly what the shipped 1024-budget run would have seen
            txt_1024 = txt if n_tok <= SHIPPED_BUDGET else tok.decode(tids[:SHIPPED_BUDGET])
            allowed = _allowed_for(r)
            pick_fixed, bad = _extract_fixed(txt, allowed)
            fout.write(json.dumps({
                "video_id": k[0], "question_id": k[1],
                "gold": str(r.get("answer_letter", "")).strip().upper(),
                "raw": txt, "allowed": allowed,
                "pick": _extract_legacy(txt),            # bare pick == LEGACY on this family
                "pick_legacy": _extract_legacy(txt),
                "pick_at1024": _extract_legacy(txt_1024),
                "pick_fixed": pick_fixed,
                "pick_strict": _extract_strict(txt, allowed),
                "bad_letter": bad,
                "n_tokens": n_tok,
                "finish_reason": fr_reason,
                "trunc_at_shipped": bool(n_tok > SHIPPED_BUDGET),
                "trunc_at_budget": bool(fr_reason == "length"),
            }) + "\n")
        fout.flush()
        print(f"  {min(c0+chunk, len(todo))}/{len(todo)}", flush=True)
    fout.close()

    # parent's accuracy block, unchanged, so the shipped print format still matches
    picks = {}
    for line in open(outp):
        r = json.loads(line); picks[(r["video_id"], r["question_id"])] = r
    n = cj = cm = cp = orc = 0
    for k, r in recs.items():
        p = picks.get(k)
        if not p or not p.get("pick"):
            continue
        g = str(r.get("answer_letter", "")).strip().upper()
        L = [t["prediction"].strip().upper() for t in r["trajectories"] if str(t.get("prediction", "")).strip()]
        if not L:
            continue
        n += 1
        cj += int(p["pick"] == g); cm += int(Counter(L).most_common(1)[0][0] == g)
        cp += L.count(g) / len(L); orc += int(g in set(L))
    if n:
        print(f"\n=== XFAMILY JURY {tag} model={model} -- {exp.split('/')[-1]} / {bench} @{max_tokens} ===")
        print(f"n={n}  pass@1={cp/n:.4f}  majority={cm/n:.4f}  XF_JURY={cj/n:.4f}  oracle={orc/n:.4f}")
        print(f"(shipped exp140 @1024: n=502 pass@1=0.6976 majority=0.7291 XF_JURY=0.6494 oracle=0.8725)")

    _derive(outp, shipped, order)


if __name__ == "__main__":
    main()
