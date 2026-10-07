import json, collections, os
CF={"Counterfactual Reasoning","Counterintuitive Comprehension","Causal Reasoning"}
ROOT=os.environ["VRITS_ROOT"] + "/experiments"
ARMS={116:"imagine_edit_27b_vmmev2_cf",117:"imagine_generate_27b_vmmev2_cf",118:"imagine_control_27b_vmmev2_cf"}
BASE={"Causal Reasoning":0.300,"Counterfactual Reasoning":0.0625,"Counterintuitive Comprehension":0.2317,"ALL":0.2393}
def analyze(eid):
    p=f"{ROOT}/{eid}_{ARMS[eid]}/eval/video-mme-v2/step_000000/predictions.jsonl"
    if not os.path.exists(p): return None
    tot=collections.Counter(); cor=collections.Counter(); nimg=0; n=0; c=0
    for line in open(p):
        try: r=json.loads(line)
        except: continue
        tt=r.get("task_type","")
        if tt not in CF: continue
        g=str(r.get("answer_letter","")).strip().upper(); pr=str(r.get("prediction","")).strip().upper()
        tot[tt]+=1; ok=int(pr==g); cor[tt]+=ok; n+=1; c+=ok
        if any(tc.get("type")=="imagine" for tc in r.get("tool_calls",[])): nimg+=1
    return {"n":n,"acc":c/n if n else 0,"by":{tt:(cor[tt]/tot[tt] if tot[tt] else 0,tot[tt]) for tt in CF},"imag_rate":nimg/n if n else 0}
res={e:analyze(e) for e in ARMS}
print("\n================ IMAGINE-LOOP CF PILOT — RESULTS ================")
print(f"{'arm':<26}{'n':>5}{'ALL-CF':>9}{'vs base':>9}{'vs ctrl':>9}{'imag%':>7}")
ctrl=res[118]["acc"] if res[118] else None
for e in (116,117,118):
    r=res[e]
    if not r: print(f"{ARMS[e]:<26}  <no predictions yet>"); continue
    vb=r["acc"]-BASE["ALL"]; vc=(r["acc"]-ctrl) if (ctrl is not None and e!=118) else 0.0
    print(f"{ARMS[e]:<26}{r['n']:>5}{r['acc']:>9.4f}{vb:>+9.4f}{('  --' if e==118 else format(vc,'+9.4f'))}{r['imag_rate']*100:>6.0f}%")
print(f"{'zeroshot CF baseline(exp5)':<26}{280:>5}{BASE['ALL']:>9.4f}")
print("\nper-subtype (acc, n) [baseline in brackets]:")
for e in (116,117,118):
    r=res[e]
    if not r: continue
    print(f"  {ARMS[e]}:")
    for tt in sorted(CF):
        a,nn=r["by"][tt]; print(f"    {tt:<32} {a:.4f} (n={nn})  [base {BASE[tt]:.4f}]")
print("\nDECISION: scale only if edit AND/OR generate beat BOTH control AND baseline(0.2393).")
