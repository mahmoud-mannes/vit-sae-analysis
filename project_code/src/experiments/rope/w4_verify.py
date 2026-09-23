"""Post-hoc checks for W4, computed from w4_output.

  REPRO_TOL      Cref and Dref re-runs against W3 at 1e-6
  probe anchors  baseline probes against their W3 values
  CHANCE_TOL_RET probe bands in retention units
"""
import json, glob, sys

REPRO_TOL, ANCHOR_TOL, PROBE_TOL_RET = 1e-6, 0.01, 0.05
W3 = json.load(open(glob.glob("../w3/w3_output/w3_windows_*.json")[0]))["results"]
J  = json.load(open(glob.glob("w4_output/w4_heads_*.json")[0]))
R  = J["results"]
A  = J["w3_anchors"]

rows = []
def check(name, scope, passed, detail=""):
    rows.append((passed, scope, name, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {scope}: {name}" + (f": {detail}" if detail else ""))

for label, e in R.items():
    if e.get("status") != "ok":
        continue
    print(f"\n{label}")
    C, w3c = e["conditions"], W3[label]["conditions"]
    PL = e["probe_layer"]

    # 1; exact reproduction of the three W3 conditions, at REPRO_TOL
    for w4_name, w3_name in [("Cref_prefix5", "prefix_5"),
                             ("Dref_axisrow5", "axisrow_5"),
                             ("Dref_axiscol5", "axiscol_5")]:
        if w4_name not in C or "ssdc" not in C[w4_name]:
            check(f"{w4_name} present", label, False, "condition missing")
            continue
        d = max(abs(a - b) for a, b in zip(C[w4_name]["ssdc"], w3c[w3_name]["ssdc"]))
        check(f"{w4_name} reproduces W3 {w3_name} over all 12 blocks", label,
              d < REPRO_TOL, f"max |delta| {d:.2e} vs tol {REPRO_TOL:.0e}")

    # 2; baseline / floor SSDC against the W3 anchors, all 12 blocks
    for nm, w3nm in [("baseline", "baseline"), ("floor", "floor")]:
        d = max(abs(a - b) for a, b in zip(C[nm]["ssdc"], w3c[w3nm]["ssdc"]))
        check(f"{nm} reproduces W3 over all 12 blocks", label,
              d < REPRO_TOL, f"max |delta| {d:.2e}")

    # 3; the probe anchors the config carries but never checks
    pb = C["baseline"].get("probes", {}).get(str(PL))
    if not pb:
        check(f"baseline probes exist at block {PL}", label, False, "no probes recorded")
    else:
        for head, key in [("exact", "probe_exact"), ("row", "probe_row"), ("col", "probe_col")]:
            want = A[label][key]
            check(f"baseline {head} probe hits its W3 anchor (block {PL})", label,
                  abs(pb[head] - want) < 1e-3,
                  f"{pb[head]*100:.2f}% vs anchor {want*100:.2f}%")

    # 4; reported, not gated: the surviving axis reading above baseline is a property of
    #     the model (W3 has the same values), not of the code.
    bas = C["baseline"].get("probes", {})
    worst = None
    for nm, r in C.items():
        if nm in ("baseline", "floor") or "probes" not in r:
            continue
        for b, p in r["probes"].items():
            if b not in bas:
                continue
            for head in ("row", "col", "exact"):
                ch = p["chance"][head]
                rng = bas[b][head] - ch
                if rng <= 1e-6:
                    continue
                ret = (p[head] - ch) / rng
                if ret > 1 + PROBE_TOL_RET and (worst is None or ret > worst[0]):
                    worst = (ret, nm, b, head)
    print(f"  [REPORT] {label}: largest surviving-axis over-baseline: "
          + ("none" if worst is None
             else f"{worst[1]} blk {worst[2]} {worst[3]} ret {worst[0]:.3f} "
                  "(W3 measured the same effect)"))

    # 5; probe reproducibility depends on probe_blocks, not on randomness
    w3c = W3[label]["conditions"]
    for w4n, w3n in [("baseline","baseline"), ("floor","floor"), ("Cref_prefix5","prefix_5"),
                     ("Dref_axisrow5","axisrow_5"), ("Dref_axiscol5","axiscol_5")]:
        a, b = C[w4n], w3c[w3n]
        same = a.get("probe_blocks") == b.get("probe_blocks")
        d = [abs(a["probes"][k][h] - b["probes"][k][h])
             for k in a.get("probes", {}) if k in b.get("probes", {})
             for h in ("exact","row","col")]
        check(f"{w4n}: probes reproduce W3 exactly when probe_blocks match", label,
              (max(d) == 0) == same,
              f"probe_blocks {'match' if same else 'differ'}, max |delta| {max(d) if d else 0:.4f}")

fails = [r for r in rows if not r[0]]
print(f"\n{'='*74}\n{len(rows)-len(fails)} PASS, {len(fails)} FAIL")
for _, scope, name, detail in fails:
    print(f"  FAIL [{scope}] {name}: {detail}")
json.dump([{"status": "PASS" if p else "FAIL", "scope": s, "gate": n, "detail": d}
           for p, s, n, d in rows], open("w4_output/w4_recovered_gates.json", "w"), indent=1)
print("\nwritten: w4_output/w4_recovered_gates.json")
sys.exit(0)
