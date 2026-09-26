"""Build the CSV tables behind the paper's W7 figures and appendix tables.

Reads the W7 JSON outputs (by default the committed ones in
results/runs/imagenet1k_val/rope_interventions/w7, or $W7_OUT), the W6 outputs for the
replication column, and the causal follow-up JSONs for the patching tables. Writes
<W7_OUT>/tables/*.csv; each file starts with a '#' line that says what it holds. Needs numpy only.
"""
import csv, glob, json, os
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[4]
RUNS = REPO / "results/runs/imagenet1k_val"
W7 = Path(os.environ.get("W7_OUT", RUNS / "rope_interventions/w7"))
W6 = RUNS / "rope_interventions/w6"
CAUSAL = RUNS / "causal_followups"
TABLES = W7 / "tables"
TABLES.mkdir(exist_ok=True)


def r4(x):
    return None if x is None else round(float(x), 4)


def write(name, header, rows, comment):
    with open(TABLES / name, "w", newline="") as f:
        f.write("# " + comment + "\n")
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


# ------------------------------------------------------------------ APE controls
APE = {"Supervised ViT-B": "w7ape_supervised_2026-09-24T18-12-56Z.json",
       "AugReg ViT-B": "w7ape_augreg_2026-09-24T18-12-56Z.json",
       "SAM ViT-B": "w7ape_sam_2026-09-25T18-42-10Z.json",  # rerun with SAM's final 24-feature set
       "DINO ViT-B": "w7ape_dino_2026-09-25T01-02-08Z.json",
       "DeiT-III ViT-B": "w7ape_deit3_2026-09-25T01-02-08Z.json"}
LABEL = {"intact": "intact", "embed_zero": "embedding zeroed", "embed_perm": "embedding permuted",
         "recon": "SAE reconstruction", "pos_zero": "positional zeroed",
         "rand_uniform": "uniform random zeroed", "recon_err": "SAE reconstruction + error",
         "pos_zero_err": "positional zeroed + error", "pos_mean_err": "positional mean-ablated + error",
         "pos_resample_samepos_err": "positional swapped, same patch other image + error",
         "pos_resample_diffpos_err": "positional swapped, other patch same image + error",
         "rand_matched_0_err": "matched random zeroed + error"}
rows = []
for m, f in APE.items():
    c = json.load(open(W7 / f))["conditions"]
    keys = [k for k in LABEL if k in c] + [f"rand_matched_{i}" for i in range(5) if f"rand_matched_{i}" in c]
    for k in keys:
        v, p = c[k], c[k].get("probes", {})
        pr = [r4(p[b][a]["test_at_best_val"] * 100) if b in p else None
              for b in ("2", "6", "11") for a in ("exact", "row", "col")]
        label = LABEL.get(k, k.replace("rand_matched_", "matched random zeroed, draw "))
        rows.append([m, k, label, r4(v["acc"] * 100) if "acc" in v else None]
                    + [r4(x) for x in v["ssdc"]] + pr)
    if "rand_matched_0" in c:
        s = [c[f"rand_matched_{i}"]["ssdc"] for i in range(5)]
        a = [c[f"rand_matched_{i}"].get("acc", np.nan) * 100 for i in range(5)]
        has_acc = not np.isnan(a[0])
        rows.append([m, "rand_matched_mean", "matched random zeroed, mean of 5",
                     r4(np.mean(a)) if has_acc else None] + [r4(x) for x in np.mean(s, 0)] + [None] * 9)
        rows.append([m, "rand_matched_sd", "matched random zeroed, s.d. of 5",
                     r4(np.std(a, ddof=1)) if has_acc else None] + [r4(x) for x in np.std(s, 0, ddof=1)] + [None] * 9)
write("app_ape_controls.csv",
      ["model", "condition", "label", "top1_pct"] + [f"ssdc_b{i}" for i in range(12)]
      + [f"probe_{a}_b{b}_pct" for b in (2, 6, 11) for a in ("exact", "row", "col")], rows,
      "W7 APE controls. SAE edits at the residual entering block 2; '+ error' adds the SAE reconstruction "
      "error back so only edited latents change. SSDC under RPI entering each block (1,000 images). Probes: "
      "linear, epoch chosen on a validation split, test accuracy (%), at blocks 2/6/11 (1,000 images). Top-1: "
      "supervised and DeiT-III 5,000 images, AugReg and SAM 1,000; DINO has no classifier. Matched random = "
      "latents matched to the positional set on log firing rate and log activation mass (5 draws).")

d = json.load(open(W7 / APE["DeiT-III ViT-B"]))["conditions"]
write("fig_depth_deit3.csv", ["model", "encoding", "condition"] + [f"b{i}" for i in range(12)],
      [["DeiT-III ViT-B", "APE", "intact"] + [r4(x) for x in d["intact"]["ssdc"]],
       ["DeiT-III ViT-B", "APE", "embedding zeroed"] + [r4(x) for x in d["embed_zero"]["ssdc"]]],
      "DeiT-III ViT-B (deit3_base_patch16_224.fb_in1k, same DeiT-III recipe as naver ViT-B), SSDC under RPI "
      "entering each block, 1,000 images.")

# ------------------------------------------------------------------ RoPE heads
ROPE = {"naver ViT-B": ("w7rope_R12_naver_2026-09-24T17-54-52Z.json", "R12_naver_correct.npz", "naver"),
        "naver ViT-S": ("w7rope_R12_small_2026-09-24T22-49-53Z.json", "R12_small_correct.npz", "small"),
        "reg1-gap ViT-B": ("w7rope_R12_reg1_gap_MERGED.json", None, "reg1_gap")}
hrows, crows = [], []
for m, (f, npz, tag) in ROPE.items():
    d = json.load(open(W7 / f))
    c, r1 = d["conditions"], d.get("top_rebuilder")
    w, b = c["window_identity"], c["baseline"]
    for k in ["baseline", "floor_identity", "floor_shuffled_s0", "floor_shuffled_s1", "window_identity", "window_shuffled"]:
        crows.append([m, k, r4(c[k]["acc"] * 100), r4(c[k]["ssdc"][4]), r4(c[k]["ssdc"][11])])
    old = json.load(open(glob.glob(str(W6 / f"w6_partA_{tag}_*.json"))[0]))["by_res"]["224"]["conditions"]
    oldw = old["Cref"]["acc"]["clean"] * 100
    oldcost = {}
    for k, v in old.items():
        for pre in ("regen_r1_h", "regen_r2_h", "random_h"):
            if k.startswith(pre):
                oldcost[int(k[len(pre):])] = oldw - v["acc"]["clean"] * 100
    P = np.load(W7 / "perimage" / npz) if npz else None
    heads = sorted(int(k[5:]) for k in c if k.startswith("win_h") and "shuffled" not in k)
    costs = {h: (w["acc"] - c[f"win_h{h}"]["acc"]) * 100 for h in heads}
    order = sorted(heads, key=lambda h: -costs[h])
    for h in heads:
        se = None
        if P is not None and f"win_h{h}" in P.files and "window_identity" in P.files:
            x = P["window_identity"].astype(float) - P[f"win_h{h}"].astype(float)
            se = x.std(ddof=1) / np.sqrt(len(x)) * 100
        hrows.append([m, h, int(h == r1), order.index(h) + 1, r4(costs[h]), r4(se),
                      r4((b["acc"] - c[f"int_h{h}"]["acc"]) * 100),
                      r4(w["ssdc"][11] - c[f"win_h{h}"]["ssdc"][11]),
                      r4(b["ssdc"][11] - c[f"int_h{h}"]["ssdc"][11]), r4(oldcost.get(h))])
write("app_rope_heads.csv",
      ["model", "head", "is_top_rebuilder", "rank_by_window_top1_cost", "window_top1_cost_pp",
       "window_cost_paired_se_pp", "intact_top1_cost_pp", "window_ssdc_b11_cost", "intact_ssdc_b11_cost",
       "earlier_run_window_top1_cost_pp"], hrows,
      "W7 all-heads test (5,000 images, 224px). window = rotations identity in blocks 0-5; cost = top-1 of that "
      "window minus top-1 with head h's rotations also identity in blocks 6-11. intact = head h's rotations "
      "identity in blocks 6-11 of the unmodified model. Paired SE over images; reg1-gap per-image data lost "
      "(overwritten in a rerun) so SE is blank. earlier_run = W6 run (3 random + 2 ranked heads only).")
write("app_rope_conditions.csv", ["model", "condition", "top1_pct", "ssdc_b4", "ssdc_b11"], crows,
      "W7 reference conditions. floor_shuffled = every rotation replaced by the rotation of a fixed random "
      "permutation of patch positions (two seeds); window_shuffled = the same in blocks 0-5 only.")

# ------------------------------------------------------------------ border and prefix
C3 = {"naver ViT-B": "w7rope_C3_naver_2026-09-24T22-49-53Z.json",
      "naver ViT-S": "w7rope_C3_small_2026-09-24T22-49-53Z.json",
      "reg1-gap ViT-B": "w7rope_C3_reg1_gap_2026-09-25T01-38-38Z.json",
      "DINOv3 ViT-B": "w7rope_C3_DINOv3_2026-09-25T01-38-38Z.json"}
brows, srows = [], []
for m, f in C3.items():
    d = json.load(open(W7 / f))
    for cond in ("intact", "prefix_block"):
        for b, rb in d[cond].items():
            for ax in ("row", "col"):
                for dist, a in sorted(rb[ax]["acc_by_border_distance"].items(), key=lambda t: int(t[0])):
                    brows.append([m, cond, int(b), ax, int(dist), r4(a * 100)])
            srows.append([m, cond, int(b), r4(rb["exact"]["test_at_best_val"] * 100),
                          r4(rb["row"]["test_at_best_val"] * 100), r4(rb["col"]["test_at_best_val"] * 100)])
    srows.append([m, "prefix_block_top1", None, r4(d["prefix_block_acc"] * 100) if d.get("prefix_block_acc") else None, None, None])
    srows.append([m, "prefix_block_ssdc_b0_to_b11", None, " ".join(f"{x:.3f}" for x in d["prefix_block_ssdc"]), None, None])
    srows.append([m, "num_prefix_tokens", None, d["num_prefix_tokens"], None, None])
write("app_border.csv", ["model", "condition", "block", "axis", "distance_to_border", "probe_acc_pct"], brows,
      "W7 C3: row/column probe test accuracy (val-selected epoch) split by the patch's distance to the nearest "
      "image border along that axis (0 = edge row/column, 6 = centre). condition prefix_block = attention from "
      "patch queries to class/register tokens masked in every block. 1,000 images, chance 7.1%.")
write("app_border_summary.csv", ["model", "condition", "block", "exact_pct", "row_pct", "col_pct"], srows,
      "W7 C3 summary; top-1 (5,000 images) and SSDC (1,000 images) under prefix blocking.")

# ------------------------------------------------------------------ patching (causal follow-ups)
rows = []
for m in ("ape", "rope"):
    for s in (0, 1):
        d = json.load(open(CAUSAL / f"final_layer_activation_patching_{m}_seed{s}.json"))
        cl, rp = d["baselines"]["clean"], d["baselines"]["rpi"]
        for k, v in d["patches"].items():
            comp, L = k.split("_L")
            if comp == "residual":
                continue
            rec = (v["scores"][11] - rp["scores"][11]) / (cl["scores"][11] - rp["scores"][11])
            rows.append([m, s, comp, int(L), r4(rec), r4(v["scores"][11]), r4(v["accuracy"] * 100),
                         r4(rp["accuracy"] * 100), r4(cl["accuracy"] * 100)] + [r4(x) for x in v["scores"]])
write("app_patching.csv",
      ["model", "seed", "component", "layer", "recovery_final", "patched_ssdc_final", "patched_top1_pct",
       "rpi_top1_pct", "clean_top1_pct"] + [f"patched_ssdc_b{i}" for i in range(12)], rows,
      "Activation patching (supervised ViT-B = ape, naver ViT-B = rope; 512 images, 2 seeds). Each attention "
      "or MLP output of the normal-order run is copied into the RPI run. recovery = (patched - RPI)/(clean - RPI) "
      "SSDC at the final readout. These experiments read the OUTPUT of block b.")
rows = []
for m in ("ape", "rope"):
    for s in (0, 1):
        d = json.load(open(CAUSAL / f"final_layer_zero_ablation_{m}_seed{s}.json"))
        b = d["baselines"]
        for k, v in d["ablations"].items():
            comp, L = k.split("_L")
            rows.append([m, s, comp, int(L), r4(v["rpi"]["final"] - b["rpi"]["final"]),
                         r4((v["clean"]["accuracy"] - b["clean"]["accuracy"]) * 100)])
        p = json.load(open(CAUSAL / f"peak_layer_zero_ablation_{m}_seed{s}.json"))
        pb = p["baselines"]["rpi"]["ssdc_by_layer"]
        for k, v in p["ablations"].items():
            comp, L = k.split("_L")
            rows.append([m, s, comp + "_peakreadout", int(L), None, None]
                        + [r4(v["rpi"]["ssdc_by_layer"][q] - pb[q]) for q in sorted(pb)])
write("app_zero_ablation.csv",
      ["model", "seed", "component", "layer", "delta_rpi_ssdc_final", "delta_clean_top1_pp",
       "delta_rpi_ssdc_readout4", "delta_rpi_ssdc_readout5"], rows,
      "Zero-ablation of one attention or MLP output (residual kept), 512 images, 2 seeds. delta = ablated minus "
      "baseline. '_peakreadout' rows read SSDC at the output of blocks 4 and 5.")
rows = []
for m in ("ape", "rope"):
    for s in (0, 1):
        d = json.load(open(CAUSAL / f"position_alignment_{m}_seed{s}.json"))
        for q in d["readout_layers"]:
            q = str(q)
            for bn in ("clean", "rpi"):
                rows.append([m, s, "baseline", bn, q, r4(d["baselines"][bn]["readout_layers"][q]),
                             r4(d["baselines"][bn]["accuracy"] * 100)])
            for sp, cc in d["controls"].items():
                for cond in ("same_image_aligned", "different_image_deranged", "same_image_misaligned", "mean_donor"):
                    v = cc[cond]
                    rows.append([m, s, sp, cond, q, r4(v["readout_layers"][q]), r4(v["accuracy"] * 100)])
write("app_alignment.csv",
      ["model", "seed", "patched_component", "condition", "readout_block_output", "ssdc", "top1_pct"], rows,
      "Alignment controls for patching (512 images, 2 seeds). Donors: same image aligned; different image, "
      "positions aligned; same image with positions shuffled; dataset mean.")

# ------------------------------------------------------------------ gate report
rows = []
for f in sorted(W7.glob("w7rope_gates_*.json")):
    for g in json.load(open(f)):
        rows.append([f.name, g["gate"], g["scope"], g["status"], g.get("detail", "")])
write("w7_gate_report.csv", ["run", "gate", "scope", "status", "detail"], rows,
      "Implementation checks run at the start of each W7 RoPE run.")
