"""Post-hoc summary and checks for W6, computed from w6_all_*.json.

1. Part A: paired regen vs random, and regen_r1 vs regen_r2.
2. Part B: block-11 SSDC noise for the three W5-added models.
3. Part C: measured probe noise.
4. Part D: spread of stage B ratios over random draws, on naver and DINOv3.
"""
import json, glob
import numpy as np

J = json.load(open(glob.glob("w6_output/w6_all_*.json")[0]))

print("1. PART A: top-1 cost, paired vs random pool, and r1-vs-r2 (quadrature SE bound)\n")
for nm, e in J["part_A"].items():
    d = e["draw"]
    print(f"--- {nm}  regen r1=h{d['regen_rank1']} (A rank {d['regen_rank1_A']})  "
          f"r2=h{d['regen_rank2']} (A rank {d['regen_rank2_A']})")
    for res, blk in e["by_res"].items():
        p = blk["paired"]
        k1 = [k for k in p if k.startswith("regen_r1")][0]
        k2 = [k for k in p if k.startswith("regen_r2")][0]
        r1, r2 = p[k1], p[k2]
        diff = r1["extra_cost_vs_random_pts"] - r2["extra_cost_vs_random_pts"]
        se = (r1["se_pts"]**2 + r2["se_pts"]**2) ** 0.5
        print(f"  {res}: r1 {r1['extra_cost_vs_random_pts']:+.2f}+-{r1['se_pts']:.2f}  "
              f"r2 {r2['extra_cost_vs_random_pts']:+.2f}+-{r2['se_pts']:.2f}  "
              f"| r1-r2 {diff:+.2f} +- {se:.2f} (quad. bound) = {diff/se:.1f} sigma (conservative)")
    print()

print("2. PART B: block-11 SSDC noise (single-condition)\n")
for nm, e in J["part_B"].items():
    b = e["by_block"]["11"]
    print(f"  {nm:10s} boot {b['boot_sd']:.5f}  seed {b['seed_sd']:.5f}  combined {b['combined_sd']:.5f}")

print("\n3. PART C: measured probe noise vs W5's 0.39pp proxy\n")
for nm, e in J["part_C"].items():
    for cond, blocks in e["conditions"].items():
        for b, heads in blocks.items():
            row = "  ".join(f"{h} {heads[h]['combined_sd']*100:.3f}pp" for h in ("exact", "row", "col"))
            print(f"  {nm:10s} {cond:16s} blk{b:>3s}  {row}")

print("\n4. PART D: the spread behind W4's single-draw ratios\n")
for nm, e in J["part_D"].items():
    rows = e["rows"]
    ks = sorted({r["k"] for r in rows})
    for k in ks:
        top = [r for r in rows if r["k"] == k and r["arm"] == "top"][0]
        rnd = [r for r in rows if r["k"] == k and r["arm"] != "top"]
        top_loss = 1 - top["retention"]
        print(f"  {nm:8s} k={k}: top loss {top_loss:.4f} (heads {top['heads']})")
        for r in rnd:
            loss = 1 - r["retention"]
            ratio = top_loss / loss if loss > 1e-9 else float("inf")
            print(f"           random {r['heads']}  overlap {r['overlap']}  loss {loss:.4f}  ratio {ratio:.1f}x")
