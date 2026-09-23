"""Post-hoc checks for W5, computed from the per-part JSONs.

1. SSDC anchors against W3 and W4 over all twelve blocks, at REPRO_TOL.
2. Hybrid: stage A against both_floor, and validity of the axis conditions.
3. Stage D labels with a magnitude floor (max drop above k x combined probe SD).
4. Part C: knockout cost on top-1 at both resolutions.
"""
import json, glob, sys, numpy as np
from scipy.stats import spearmanr
OUT="w5_output"; REPRO_TOL=1e-6; K_FLOOR=3.0
W3=json.load(open(glob.glob("../w3/w3_output/w3_windows_*.json")[0]))["results"]
W4=json.load(open(glob.glob("../w4/w4_output/w4_heads_*.json")[0]))["results"]
PA={l:json.load(open(p)) for l,p in [(p.split("w5_partA_")[1].rsplit("_2026",1)[0],p) for p in sorted(glob.glob(f"{OUT}/w5_partA_*.json"))]}
PC={l:json.load(open(p)) for l,p in [(p.split("w5_partC_")[1].rsplit("_2026",1)[0],p) for p in sorted(glob.glob(f"{OUT}/w5_partC_*.json"))]}
full={"RoPE · naver":"naver","RoPE · DINOv3":"DINOv3"}
rows=[]; rep={}
def check(name,scope,ok,detail="",kind="gate"):
    st=("PASS" if ok else "FAIL") if kind=="gate" else ("CONFIRMED" if ok else "not seen")
    rows.append({"gate":name,"scope":scope,"kind":kind,"status":st,"detail":detail})
    print(f"  [{st}] {scope}: {name}" + (f": {detail}" if detail else ""))

print("1. SSDC anchors, all twelve blocks")
for lab,short in full.items():
    if short not in PA: continue
    C=PA[short]["conditions"]
    for w5n,w3n,w4n in [("baseline","baseline","baseline"),("floor","floor","floor"),
                        ("Cref_prefix5","prefix_5","Cref_prefix5"),("Dref_axisrow5","axisrow_5","Dref_axisrow5"),
                        ("Dref_axiscol5","axiscol_5","Dref_axiscol5")]:
        d3=max(abs(a-b) for a,b in zip(C[w5n]["ssdc"],W3[lab]["conditions"][w3n]["ssdc"]))
        d4=max(abs(a-b) for a,b in zip(C[w5n]["ssdc"],W4[lab]["conditions"][w4n]["ssdc"]))
        check(f"{w5n} reproduces W3 and W4 over 12 blocks",short,max(d3,d4)<REPRO_TOL,f"W3 {d3:.2e}  W4 {d4:.2e}")
    for w5n,w4n in [("axis_row","A_head00")]: pass

print("\n2. rope_ape hybrid")
if "rope_ape" in PA:
    e=PA["rope_ape"]; C=e["conditions"]; PL=e["probe_layer"]; nh=e["n_heads"]
    B=C["baseline"]["ssdc"]; Fb=C["both_floor"]["ssdc"]; F0=C["floor"]["ssdc"]
    r=lambda s,b,fl:(s[b]-fl[b])/(B[b]-fl[b])
    check("rope-only floor exceeds baseline at the probe layer, so the retention range is negative","rope_ape",
          B[PL]-F0[PL]<0,f"baseline {B[PL]:.4f} floor {F0[PL]:.4f}",kind="finding")
    A=sorted(range(nh),key=lambda h:(r(C[f"A_head{h:02d}"]["ssdc"],PL,Fb),h))
    rets=[r(C[f"A_head{h:02d}"]["ssdc"],PL,Fb) for h in range(nh)]
    rep["rope_ape_stageA_both_floor"]={"ranking":A,"retention":dict(zip(map(str,range(nh)),rets))}
    print(f"    stage A vs both_floor: ranking {A}, retention {min(rets):.2f}..{max(rets):.2f} (JSON stored the inverse)")
    pb=C["baseline"]["probes"][str(PL)]
    for cond in ("axis_row","axis_col","floor"):
        p=C[cond]["probes"][str(PL)]
        check(f"{cond} leaves both axis probes at baseline: APE carries position, so the condition is uninformative here","rope_ape",
              p["row"]>0.9*pb["row"] and p["col"]>0.9*pb["col"],
              f"row {p['row']*100:.1f}% col {p['col']*100:.1f}% vs baseline {pb['row']*100:.1f}/{pb['col']*100:.1f}",kind="finding")
    curve={n:C[n]["ssdc"] for n in ("baseline","floor","ape_floor","both_floor")}
    rep["rope_ape_floor_curves"]=curve
    late=max(b for b in range(12) if curve["floor"][b]>=curve["baseline"][b])
    print(f"    rope-off SSDC >= baseline through block {late}; rope-only (ape_floor) reaches {curve['ape_floor'][11]/curve['baseline'][11]*100:.0f}% of baseline at block 11")

print("\n3. Stage D with a magnitude floor")
sd_ref={}
for short,e in PA.items():
    n=e["conditions"]["baseline"]["probes"].get("7",{}).get("noise")
    if n: sd_ref[short]=float(np.mean([n[h]["combined_sd"] for h in ("row","col")]))*100
proxy=float(np.mean(list(sd_ref.values()))) if sd_ref else 0.35
labels={}
for short,e in PA.items():
    if short=="rope_ape": print("    rope_ape: skipped, axis conditions uninformative (see 2)"); continue
    C=e["conditions"]; bl=range(7,12); rr=C["Dref_axisrow5"]["probes"]; rc=C["Dref_axiscol5"]["probes"]
    sd=sd_ref.get(short,proxy); src="measured" if short in sd_ref else "proxy"
    out=[]
    for d in e["stageC_ranking"][:e["d_top"]]:
        h=d["head"]; pr=C[f"D_row_head{h:02d}"]["probes"]; pc=C[f"D_col_head{h:02d}"]["probes"]
        a=np.mean([(rr[str(b)]["row"]-pr[str(b)]["row"])*100 for b in bl]); b_=np.mean([(rr[str(b)]["col"]-pr[str(b)]["col"])*100 for b in bl])
        c=np.mean([(rc[str(b)]["row"]-pc[str(b)]["row"])*100 for b in bl]); d_=np.mean([(rc[str(b)]["col"]-pc[str(b)]["col"])*100 for b in bl])
        rule="ROW" if a>2*max(b_,0.01) else ("COL" if d_>2*max(c,0.01) else "mixed")
        mag=max(a,d_); lab=rule if mag>=K_FLOOR*sd else "below floor"
        out.append({"head":h,"row_win_row":a,"row_win_col":b_,"col_win_row":c,"col_win_col":d_,"rule":rule,"max_drop_pp":mag,"sd_pp":sd,"label":lab})
    labels[short]=out
    print(f"    {short:9s} SD {sd:.2f}pp ({src}):  " + "  ".join(f"h{o['head']}:{o['label']}({o['max_drop_pp']:.1f})" for o in out))
rep["stageD_labels_with_floor"]=labels

print("\n4. Part C, top-1 cost over Cref (pp), regenerator vs random head")
pc_rows=[]
for short,R in PC.items():
    line=f"    {short:9s} regen h{R['top_regenerator']} random h{R['random_head']}  "
    for res,blk in sorted(R["by_res"].items(),key=lambda kv:int(kv[0])):
        c=blk["conditions"]; cref=c["Cref"]["acc"]["clean"]
        rg=(cref-c["regen_ko"]["acc"]["clean"])*100; rd=(cref-c["random_ko"]["acc"]["clean"])*100
        base=c["baseline"]["acc"]["clean"]*100; win=(c["baseline"]["acc"]["clean"]-cref)*100; fl=(c["baseline"]["acc"]["clean"]-c["floor"]["acc"]["clean"])*100
        line+=f"| {res}: base {base:.1f}  window {win:+.1f}  rope-off {fl:+.1f}  regen {rg:+.2f}  random {rd:+.2f} "
        pc_rows.append({"model":short,"res":int(res),"baseline_top1":base,"window_cost":win,"rope_off_cost":fl,"regen_ko_over_Cref":rg,"random_ko_over_Cref":rd})
    print(line)
rep["partC"]=pc_rows

gates=[r_ for r_ in rows if r_["kind"]=="gate"]; fails=[r_ for r_ in gates if r_["status"]=="FAIL"]
finds=[r_ for r_ in rows if r_["kind"]=="finding"]
print(f"\ngates: {len(gates)-len(fails)} PASS, {len(fails)} FAIL   |   findings documented: {len(finds)}")
json.dump({"checks":rows,"report":rep},open(f"{OUT}/w5_recovered_analysis.json","w"),indent=1)
print(f"written: {OUT}/w5_recovered_analysis.json")
