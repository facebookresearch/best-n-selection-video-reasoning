"""Instrumented fork of verifiers/verify_jury_local.py -- measures the 1024-token
truncation exposure on the Qwen-27B SELF-JURY leg of the Video-Holmes selection
increment.

WHY THIS EXISTS
---------------
The paper's headline Video-Holmes selection increment is

    SI = LOCAL_JURY(56.4) - standalone(27.7) = +28.7

Both legs were generated with a 1024-token cap. The standalone leg has been
measured (exp 393/394/395) to be catastrophically truncated: 86.8% of its rows
hit the cap, and the corrected accuracy is +25.6 points higher. The jury leg
runs through verify_jury_local.py, which also caps at 1024
(verify_jury_local.py:137, `max_new_tokens=1024`). Zero-GPU tqdm telemetry on
`156_vholmes_jury_qwen27b_self/train.log` puts that cell at ~485 output
tokens/request with individual batch means as high as 822 and ~20% of request
mass above 512 tokens -- i.e. PARTIALLY truncated, severity unknown.

Fixing only the standalone leg would replace one budget-confounded number with
another. This run measures the jury leg at the same 8192-token budget so the SI
can be stated at a MATCHED generation budget.

WHAT IS DIFFERENT FROM THE PARENT (and what is deliberately identical)
----------------------------------------------------------------------
IDENTICAL, byte-for-byte -- these are the control surface, do NOT "improve" them:
  * `_prompt()`         -- same string, same 500-char reasoning truncation
  * `_key()`            -- same keyer (incl. the question-text fallback)
  * `_extract_legacy()` -- the parent's extractor, article-fallback bug INCLUDED
  * prompt construction -- goes through vlm._build_messages +
                           vlm._apply_chat_template, exactly as
                           VLM.generate_batch does
  * SamplingParams      -- temperature=0.0, top_p=1.0, seed=vlm.seed
  * batching            -- same chunk size, same record order, so vLLM batch
                           composition matches the shipped run

DIFFERENT:
  1. `max_tokens` defaults to 8192 (was hard-coded 1024).
  2. Calls `vlm.llm.generate(...)` directly instead of `vlm.generate_batch(...)`.
     Reason: generate_batch (methods/vlm_backend.py:140) returns only
     `(text, len(token_ids))` -- it discards `token_ids` and `finish_reason`,
     and verify_jury_local.py:138 throws even the length away as `_n`. We need
     the raw token IDs to reconstruct the shipped run. The request dicts are
     built with the same two backend helpers, so the PROMPT IS UNCHANGED.
  3. Records per row: raw text, n_tokens, finish_reason, and FOUR picks:
       pick_legacy  -- parent extractor on the full 8192-token text
       pick_at1024  -- parent extractor on detokenize(token_ids[:1024])
                       == an exact reconstruction of the shipped 1024-cap run,
                       because decoding is greedy with an identical prompt
       pick_fixed   -- marker-first, then UPPERCASE-only standalone-letter
                       fallback, clamped to this question's option letters
       pick_strict  -- marker only; abstains rather than guess
  4. Writes to THIS experiment's own eval tree. The parent computes its output
     path as `pred.parent / f"jury_local_{tag}_picks.jsonl"` and opens it in
     APPEND mode -- pointing the parent at exp149 would append to, and thereby
     corrupt, the shipped picks file that is the paper's evidence. We never
     write outside this experiment directory.

THE CONTROL GATE
----------------
`--shipped <path>` joins our rows against the shipped picks file and reports

    shipped.pick == rerun.pick_at1024

on the intersection. This is a GATE, not a diagnostic: greedy decoding at
temperature 0 with seed 42 on a byte-identical prompt means the first 1024
tokens generated under an 8192 cap ARE the shipped run's entire output. If this
does not reproduce at ~100%, the reconstruction is not faithful and no corrected
number from this run may be reported. (Expect a small residual from vLLM batch
composition nondeterminism -- the standalone re-run showed 1 mismatch in 226
gate-eligible rows.)

USAGE (matches the shipped exp156 config exactly -- read off its train.log)
    python jury_local_instrumented.py <pool_exp_dir> \
      --bench video-holmes --n-samples 1837 --frames 48 \
      --model Qwen/Qwen3.5-27B --tp 1 --dtype bfloat16 --chunk 8 \
      --max-model-len 49152 --max-tokens 8192 \
      --shipped <pool_exp_dir>/eval/video-holmes/step_000000/jury_local_Qwen3527B_f48_picks.jsonl

Resumable: re-running skips rows already in our own output file.
"""
from __future__ import annotations
import os, sys, re, json
from collections import Counter
from pathlib import Path

# --- env must be set before vllm/transformers import ---------------------------
# TRANSPLANTED VERBATIM from 393/394/395's standalone_parsefix.py, which boot
# cleanly in this venv on this cluster right now. The parent verify_jury_local.py
# pops only the proxy vars and relies on the launching shell for HF_HOME +
# offline mode; that is why the first exp397 attempt (job 10481250) died in
# ModelConfig -> get_config -> hf_api().list_repo_files() with
# FileNotFoundError on os.environ["SSL_CERT_FILE"]: with HF_HOME unset the
# snapshot was not in the default cache, so vLLM went to the network, and the
# harness exports SSL_CERT_FILE pointing at a path that does not exist here.
# Numerical safety: 393/394/395 run under exactly this block and their
# <=1024-token control gate reproduces the shipped standalone picks 20/20,
# 39/40, 166/166 -- so this block does not perturb greedy decoding.
for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
           "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
    os.environ.pop(_v, None)
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DECORD_EOF_RETRY_MAX"] = "20480"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"

CODE = os.environ["VRITS_ROOT"] + "/experiments/109_bon8_verifier_zeroshot_27b_vmmev2/code"
CACHE = os.environ["VRITS_DATA"] + "/video_cache"
sys.path.insert(0, CODE)

# The budget the shipped run used. pick_at1024 reconstructs that run.
SHIPPED_BUDGET = 1024

# ---------------------------------------------------------------- extractors
# TRANSPLANTED VERBATIM from the OTHER SI leg
# (393/394/395 code/standalone_parsefix.py). The corrected selection increment is
#   SI_corrected = jury(this run) - standalone(393/394/395)
# so both legs MUST apply the SAME extractor definitions, or the subtraction
# silently mixes two parsing conventions and the increment is uninterpretable.
# Do not "improve" these here without changing them in 393/394/395 too.
_ANS_LEGACY = re.compile(r"answer\s*[:\-]\s*\(?([A-Ha-h])\)?", re.IGNORECASE)
# `(?![A-Za-z])` is Fix A: it stops the marker regex matching the 'A' of a
# following literal "Answer:" ("Final Answer:\n\nAnswer: F" -> legacy 'A').
_ANS_FIXED = re.compile(r"answer\s*[:\-]\s*\(?([A-Ha-h])\)?(?![A-Za-z])", re.IGNORECASE)

_ALL = "ABCDEFGH"


def _extract_legacy(t):
    """VERBATIM copy of verify_jury_local.py:_extract -- the parent's behaviour,
    article-fallback bug included. This is the CONTROL column. Do not fix it.

    The fallback scans for standalone [A-Ha-h] case-INSENSITIVELY, so on prose
    that was cut mid-sentence the last standalone letter is very often the
    indefinite article "a" -> a spurious 'A'. It also does NOT clamp to the
    question's option letters, so it can emit a letter that is not on offer.
    """
    if not t:
        return ""
    m = _ANS_LEGACY.search(t)
    if m:
        return m.group(1).upper()
    for ch in reversed(re.findall(r"\b([A-Ha-h])\b", t)):
        return ch.upper()
    return ""


def _extract_fixed(t, allowed):
    """Word-boundary-guarded, clamped to the letters this question actually has.

    Returns (pick, bad_letter). `bad_letter` is a letter that was extracted but
    does not exist as an option -- kept for the record, not scored.
    """
    if not t:
        return "", ""
    bad = ""
    for m in _ANS_FIXED.finditer(t):
        ch = m.group(1).upper()
        if ch in allowed:
            return ch, ""
        bad = bad or ch
    # Fallback: last standalone UPPERCASE option letter in the response.
    # Uppercase-only is the whole point. The parent scanned [A-Ha-h], which matches
    # the English indefinite article "a"; on a truncated reasoning chain (no answer
    # marker at all) reversed() then returns that article, so 81% of truncated rows
    # were scored "A". Option letters are always uppercase in this prompt.
    #
    # JURY-SPECIFIC CAVEAT: this fallback is still not trustworthy HERE. The jury
    # prompt quotes the other analyses' letter conclusions ("Analysis 1 (concludes
    # B): ..."), so on a truncated jury response the last uppercase letter is
    # frequently a QUOTED CANDIDATE letter rather than the verifier's own verdict.
    # pick_strict is the only column that abstains instead of guessing.
    for ch in reversed(re.findall(r"\b([A-H])\b", t)):
        if ch in allowed:
            return ch, bad
        bad = bad or ch
    return "", bad


def _extract_strict(t, allowed):
    """Marker-based only -- NO fallback. If the model never stated an answer the
    row is unparseable, and saying so is more honest than guessing from prose."""
    if not t:
        return ""
    for m in _ANS_FIXED.finditer(t):
        ch = m.group(1).upper()
        if ch in allowed:
            return ch
    return ""


def _allowed_for(rec):
    """Letters that genuinely exist for this question, from its option list.

    Same body as the standalone leg's `_allowed_for`; only the lookup differs --
    the jury reads options off the pooled record's `meta`, the standalone leg
    reads them off the benchmark sample.
    """
    opts = (rec.get("meta") or {}).get("options") or []
    letters = []
    for o in opts:
        m = re.match(r"\s*\(?([A-Ha-h])\s*[\.\):]", str(o))
        if m:
            letters.append(m.group(1).upper())
    if letters:
        return "".join(sorted(set(letters)))
    return _ALL[:len(opts)] if opts else _ALL


def _argval(flag, default):
    return type(default)(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


# ---------------------------------------------------- parent-identical pieces
def _key(video_id, question_id, question):
    # VERBATIM from verify_jury_local.py:_key. question_id may be None/""
    # (e.g. TempCompass) -> fall back to the question text so rows don't
    # collapse to one-per-video.
    qid = question_id if question_id not in (None, "") else question
    return (video_id, qid)


def _prompt(rec, n_imgs):
    # VERBATIM from verify_jury_local.py:_prompt. Byte-identical or the whole
    # comparison is void.
    q = rec.get("meta", {}).get("question", ""); opts = rec.get("meta", {}).get("options", [])
    cand = "\n".join(
        f"Analysis {i+1} (concludes {t.get('prediction','?')}): {str(t.get('reasoning',''))[:500]}"
        for i, t in enumerate(rec["trajectories"])
    )
    return (
        f"You are shown {n_imgs} frames sampled uniformly from a video. Use the FRAMES as the "
        f"ground-truth evidence.\n\nQuestion: {q}\nOptions:\n" + "\n".join(opts) +
        f"\n\n{len(rec['trajectories'])} independent analyses of this video were produced:\n{cand}\n\n"
        "Verify the analyses against what you actually see in the frames and decide which option is "
        "correct. Respond with ONLY the final answer as 'Answer: X' (a single letter)."
    )


# ------------------------------------------------------------ control gate
def _gate(outp, shipped_path):
    """shipped.pick MUST equal our pick_at1024 on the joined rows."""
    if not shipped_path or not Path(shipped_path).exists():
        print(f"[gate] no shipped file at {shipped_path!r} -- SKIPPED", flush=True)
        return
    shp = {}
    for line in open(shipped_path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        shp[(str(r.get("video_id", "")), str(r.get("question_id", "")))] = r
    n = ok = 0
    mism = []
    for line in open(outp):
        try:
            r = json.loads(line)
        except Exception:
            continue
        s = shp.get((str(r.get("video_id", "")), str(r.get("question_id", ""))))
        if s is None:
            continue
        n += 1
        a = str(s.get("pick", "")).strip().upper()
        b = str(r.get("pick_at1024", "")).strip().upper()
        if a == b:
            ok += 1
        elif len(mism) < 10:
            mism.append(((r.get("video_id"), r.get("question_id")), a, b, r.get("n_tokens")))
    print(f"\n=== CONTROL GATE: shipped.pick == rerun.pick_at1024 ===", flush=True)
    if n:
        print(f"  {ok}/{n} = {100.0*ok/n:.2f}%  (shipped rows joined: {n})", flush=True)
    else:
        print("  0 rows joined -- CANNOT GATE", flush=True)
    for k, a, b, t in mism:
        print(f"  MISMATCH {k}: shipped={a!r} rerun_at1024={b!r} n_tok={t}", flush=True)


def main():
    exp = sys.argv[1].rstrip("/")
    bench = _argval("--bench", "video-mme-v2")
    n_samples = _argval("--n-samples", 500)
    subset = _argval("--subset", "")
    n_frames = _argval("--frames", 48)
    model = _argval("--model", "Qwen/Qwen3.5-27B")
    tp = _argval("--tp", 1)
    dtype = _argval("--dtype", "bfloat16")
    chunk = _argval("--chunk", 8)
    limit = _argval("--limit", 10 ** 9)
    max_model_len = _argval("--max-model-len", 49152)
    max_tokens = _argval("--max-tokens", 8192)
    shipped = _argval("--shipped", "")
    gate_only = "--gate-only" in sys.argv

    from methods.vlm_backend import VLM, frames_to_pil
    from video_io import VideoReader
    from benchmarks import load_benchmark

    tag = model.split("/")[-1].replace(".", "").replace("-", "")[:12] + f"_f{n_frames}"
    pred = sorted(Path(exp).glob("eval/**/predictions.jsonl"))[0]

    # NEVER write into the pool experiment: the parent's `pred.parent / ..._picks.jsonl`
    # is the SHIPPED evidence file and the parent opens it in append mode.
    outdir = Path("eval") / bench / "step_000000"
    outdir.mkdir(parents=True, exist_ok=True)
    outp = outdir / f"jury_local_{tag}_instrumented.jsonl"
    print(f"[jury8k] pool={pred}\n[jury8k] out={outp.resolve()}\n"
          f"[jury8k] budget={max_tokens} (shipped was {SHIPPED_BUDGET})", flush=True)

    if gate_only:
        _gate(outp, shipped)
        return

    recs = {}
    for line in open(pred):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("trajectories"):
            recs[_key(r.get("video_id"), r.get("question_id"), (r.get("meta") or {}).get("question"))] = r
    smap = {}
    _bkw = {"subset": subset} if subset else {}
    for s in load_benchmark(bench, cache_root=CACHE, n_samples=n_samples, **_bkw):
        smap[_key(s["video_id"], s.get("meta", {}).get("question_id"), s.get("question"))] = s

    done = set()
    if outp.exists():
        for line in open(outp):
            try:
                r = json.loads(line); done.add((r["video_id"], r["question_id"]))
            except Exception:
                pass
    todo = [(k, r) for k, r in recs.items() if k not in done and k in smap][:limit]
    print(f"jury_local_instr[{tag}]: {len(recs)} recs, {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        _gate(outp, shipped)
        return

    # Same VLM config the shipped run used (read off 156_.../train.log):
    #   dtype=bfloat16, tp=1, max_model_len=49152, max_num_seqs=8, seed=42,
    #   limit_mm_per_prompt={'image': n_frames+4}
    vlm = VLM(hf_id=model, tp_size=tp, dtype=dtype, max_model_len=max_model_len,
              gpu_memory_utilization=0.92, seed=42,
              limit_mm_per_prompt={"image": n_frames + 4}, max_num_seqs=max(4, chunk))
    reader = VideoReader(cache_root=CACHE, frame_extractor="decord", frame_resize=336)

    from vllm import SamplingParams
    # Identical to what VLM.generate_batch would construct, except max_tokens.
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens, seed=vlm.seed)

    tok = None
    try:
        tok = vlm.llm.get_tokenizer()
    except Exception:
        tok = getattr(vlm.processor, "tokenizer", None)
    if tok is None:
        print("[jury8k] FATAL: no tokenizer -> cannot reconstruct pick_at1024", flush=True)
        return

    fout = open(outp, "a")
    n_trunc_shipped = n_trunc_budget = 0
    for c0 in range(0, len(todo), chunk):
        reqs, keys = [], []
        for k, r in todo[c0:c0 + chunk]:
            try:
                frames, _ = reader.load_and_sample(smap[k]["video_path"], n_frames)
                imgs = frames_to_pil(frames)
            except Exception:
                fout.write(json.dumps({"video_id": k[0], "question_id": k[1], "pick": "",
                                       "pick_legacy": "", "pick_at1024": "", "pick_fixed": "",
                                       "pick_strict": "",
                                       "gold": str(r.get("answer_letter", "")).strip().upper(),
                                       "err": "decode"}) + "\n"); fout.flush()
                continue
            # Prompt built through the SAME backend helpers generate_batch uses,
            # so the prompt string is byte-identical to the shipped run's.
            text = _prompt(r, len(imgs))
            messages = vlm._build_messages(imgs, text)
            prompt = vlm._apply_chat_template(messages)
            req = {"prompt": prompt}
            if imgs:
                req["multi_modal_data"] = {"image": imgs}
            reqs.append(req); keys.append((k, r))
        if not reqs:
            continue
        outs = vlm.llm.generate(reqs, sampling_params=sampling)
        for (k, r), o in zip(keys, outs):
            co = o.outputs[0]
            txt = co.text
            tids = list(co.token_ids)
            n_tok = len(tids)
            fr = getattr(co, "finish_reason", "")
            allowed = _allowed_for(r)

            # Reconstruct the shipped 1024-cap run: greedy + identical prompt
            # means the first 1024 tokens here ARE that run's whole output.
            if n_tok > SHIPPED_BUDGET:
                txt_1024 = tok.decode(tids[:SHIPPED_BUDGET], skip_special_tokens=True)
                n_trunc_shipped += 1
            else:
                txt_1024 = txt
            if fr == "length":
                n_trunc_budget += 1

            pick_fixed, bad = _extract_fixed(txt, allowed)
            fout.write(json.dumps({
                "video_id": k[0], "question_id": k[1],
                # `pick` keeps the parent's semantics so downstream scorers that
                # read this file behave identically to the shipped surface.
                "pick": _extract_legacy(txt),
                "pick_legacy": _extract_legacy(txt),
                "pick_at1024": _extract_legacy(txt_1024),
                "pick_fixed": pick_fixed,
                "pick_strict": _extract_strict(txt, allowed),
                "gold": str(r.get("answer_letter", "")).strip().upper(),
                "allowed": allowed,
                "bad_letter": bad,
                "n_tokens": n_tok,
                "finish_reason": fr,
                "trunc_at_shipped": bool(n_tok > SHIPPED_BUDGET),
                "trunc_at_budget": bool(fr == "length"),
                "raw": txt,
            }) + "\n")
        fout.flush()
        print(f"  {min(c0+chunk, len(todo))}/{len(todo)}  "
              f"[would-truncate-at-{SHIPPED_BUDGET}: {n_trunc_shipped}, "
              f"still-truncated-at-{max_tokens}: {n_trunc_budget}]", flush=True)
    fout.close()

    # ---------------------------------------------------------------- score
    picks = {}
    for line in open(outp):
        try:
            r = json.loads(line)
        except Exception:
            continue
        picks[(r["video_id"], r["question_id"])] = r

    for col in ("pick_legacy", "pick_at1024", "pick_fixed", "pick_strict"):
        n = cj = cm = orc = 0
        cp = 0.0
        pa = 0
        for k, r in recs.items():
            p = picks.get(k)
            if not p:
                continue
            g = str(r.get("answer_letter", "")).strip().upper()
            L = [t["prediction"].strip().upper() for t in r["trajectories"]
                 if str(t.get("prediction", "")).strip()]
            if not L:
                continue
            n += 1
            pick = str(p.get(col, "")).strip().upper()
            cj += int(pick == g); cm += int(Counter(L).most_common(1)[0][0] == g)
            cp += L.count(g) / len(L); orc += int(g in set(L))
            pa += int(pick == "A")
        if n:
            print(f"\n=== LOCAL JURY [{col}] {tag} -- {exp.split('/')[-1]} / {bench} ===")
            print(f"n={n}  pass@1={cp/n:.4f}  majority={cm/n:.4f}  "
                  f"LOCAL_JURY={cj/n:.4f}  oracle={orc/n:.4f}  pickA={100.0*pa/n:.2f}%")

    _gate(outp, shipped)


if __name__ == "__main__":
    main()
