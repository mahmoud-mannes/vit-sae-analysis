# ---- cell 2 ----
import json, re, io, os, sys, math, warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
N_BLOCKS      = 12     # ViT-B/16 depth. Every condition must have exactly this many values.
DAMAGE_GATE   = 0.30   # recovery is not reported below 30% damage at the site
MIN_BASELINE  = 0.05   # Below this, a denominator is too small to divide by. Ratio -> NaN.
NOISE_SIGMA_K = 3.0    # An effect must exceed this many noise-spreads to count as real.
OVERSHOOT_TOL = 1.10   # Retention above this is flagged as overshoot, not recovery.

RUN_STAMP = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
print("W1-A analysis |", RUN_STAMP)
print("numpy", np.__version__, "| pandas", pd.__version__)

# ---- cell 4 ----
UPLOADED = {}

try:
    from google.colab import files as colab_files
    IN_COLAB = True
except Exception:
    IN_COLAB = False

if IN_COLAB:
    print("Select the ablation JSON file(s). You can pick both at once.\n")
    up = colab_files.upload()
    for fname, payload in up.items():
        UPLOADED[fname] = json.loads(payload.decode("utf-8"))
        print(f"  loaded {fname}  ({len(payload):,} bytes)")
else:
    runs = Path(__file__).resolve().parents[4] / "results" / "runs" / "SAE_20k_images_analysis"
    found = [runs / "original_results" / "feature_ablation_residual.json",
             runs / "second_seed" / "feature_ablation_residual_second_seed.json"]
    for path in found:
        with open(path) as fh:
            UPLOADED[path.name] = json.load(fh)
        print(f"  loaded {path.name}")

if not UPLOADED:
    raise RuntimeError("No files were uploaded: re-run this cell.")

print(f"\n{len(UPLOADED)} file(s) in memory.")

# ---- cell 6 ----
FLAGS = []

def flag(severity, scope, message):
    '''Record a data-quality issue. severity in ERROR / WARN / NOTE.'''
    FLAGS.append({"severity": severity, "scope": scope, "message": message})

def conds_of(entries):
    out = {}
    for e in entries:
        name = e.get("experiment_name")
        vals = e.get("results")
        if name is None or vals is None:
            flag("ERROR", "loader", f"malformed entry: {str(e)[:80]}")
            continue
        if name in out:
            flag("ERROR", "loader", f"duplicate condition name '{name}': keeping the first")
            continue
        out[name] = [float(v) for v in vals]
    return out

def classify_seed(doc):
    '''Seed 2 (DINOv3) has a RoPE baseline near the ceiling; seed 1 does not.'''
    for key, entries in doc.items():
        if not key.startswith("RoPE"):
            continue
        c = conds_of(entries)
        base = c.get("no_SAE_baseline") or c.get("SAE_baseline")
        if base is None:
            continue
        mid = np.nanmean(base[4:9])
        return "seed2" if mid > 0.85 else "seed1"
    return "unknown"

MODEL_LABELS = {
    ("seed1", "APE"):  "APE · google ViT",
    ("seed1", "RoPE"): "RoPE · naver",
    ("seed2", "APE"):  "APE · DINOv1",
    ("seed2", "RoPE"): "RoPE · DINOv3",
}

RUNS = {}
for fname, doc in UPLOADED.items():
    seed = classify_seed(doc)
    if seed == "unknown":
        flag("ERROR", fname, "could not classify this file as seed 1 or seed 2: skipped")
        continue
    for key, entries in doc.items():
        m = re.match(r"(APE|RoPE)_layer_(\d+)$", key)
        if not m:
            flag("WARN", fname, f"unrecognised top-level key '{key}': skipped")
            continue
        family, layer = m.group(1), int(m.group(2))
        label = MODEL_LABELS.get((seed, family), f"{seed}·{family}")
        c = conds_of(entries)

        for name, vals in c.items():
            if len(vals) != N_BLOCKS:
                flag("ERROR", label, f"'{name}' has {len(vals)} blocks, expected {N_BLOCKS}")
            if not all(np.isfinite(vals)):
                flag("ERROR", label, f"'{name}' contains non-finite values")
        for req in ("SAE_baseline", "no_SAE_baseline"):
            if req not in c:
                flag("ERROR", label, f"missing required denominator '{req}'")

        if label in RUNS:
            flag("ERROR", label, "duplicate run: a later file overwrote an earlier one")
        RUNS[label] = {"seed": seed, "family": family, "layer": layer,
                       "source_file": fname, "conds": c}

order = ["APE · google ViT", "RoPE · naver", "APE · DINOv1", "RoPE · DINOv3"]
RUNS = {k: RUNS[k] for k in order if k in RUNS} | {k: v for k, v in RUNS.items() if k not in order}

inv = pd.DataFrame([{
    "run": k, "seed": v["seed"], "family": v["family"],
    "intervention_layer": v["layer"], "n_conditions": len(v["conds"]),
    "source_file": v["source_file"],
} for k, v in RUNS.items()])
print("Loaded runs\n")
print(inv.to_string(index=False))
print(f"\nErrors so far: {sum(f['severity']=='ERROR' for f in FLAGS)}")

# ---- cell 7 ----
# condition inventory
def dose_of(name):
    m = re.match(r"SAE_(\d+)_features_ablated$", name)
    return int(m.group(1)) if m else None

def random_dose_of(name):
    m = re.match(r"(?:SAE_)?(\d+)_random_features_ablated$", name)
    return int(m.group(1)) if m else None

def kind_of(name):
    if name in ("SAE_baseline", "no_SAE_baseline"):
        return "baseline"
    if random_dose_of(name) is not None:
        return "random_control"
    if dose_of(name) is not None:
        return "ablation"
    return "unknown"

rows = []
for label, R in RUNS.items():
    for name in R["conds"]:
        k = kind_of(name)
        if k == "unknown":
            flag("WARN", label, f"unclassified condition '{name}': excluded from analysis")
        rows.append({"run": label, "condition": name, "kind": k,
                     "dose": dose_of(name) if k == "ablation" else random_dose_of(name)})
inventory = pd.DataFrame(rows)

for label, R in RUNS.items():
    sub = inventory[(inventory.run == label)]
    if not (sub.kind == "random_control").any():
        flag("WARN", label, "no random-feature control present: dose effects are uncontrolled")
    doses = sorted(sub[sub.kind == "ablation"].dose.dropna().astype(int).tolist())
    print(f"{label:20s} layer {R['layer']:>2}  doses {doses}  "
          f"random {sorted(sub[sub.kind=='random_control'].dose.dropna().astype(int).tolist())}")

# ---- cell 9 ----
noise_rows = []
for label, R in RUNS.items():
    L, c = R["layer"], R["conds"]
    if L == 0:
        flag("WARN", label, "intervention at block 0: no pre-intervention blocks, no noise estimate")
        R["noise_median"] = np.nan
        R["noise_max"] = np.nan
        continue
    arr = np.array([v for v in c.values()])
    spreads = []
    for b in range(L):
        col = arr[:, b]
        spread = float(np.nanmax(col) - np.nanmin(col))
        spreads.append(spread)
        noise_rows.append({"run": label, "block": b, "spread": spread,
                           "min": float(np.nanmin(col)), "max": float(np.nanmax(col)),
                           "mean_abs_level": float(np.nanmean(np.abs(col)))})
    R["noise_median"] = float(np.median(spreads))
    R["noise_max"] = float(np.max(spreads))

noise_detail = pd.DataFrame(noise_rows)
noise_summary = (noise_detail.groupby("run")
                 .agg(blocks_checked=("block", "count"),
                      noise_median=("spread", "median"),
                      noise_max=("spread", "max"))
                 .reindex([k for k in RUNS if k in set(noise_detail.run)]))

print("Run-to-run noise, measured on blocks that the intervention cannot reach\n")
print(noise_summary.to_string())
print("\nPer-block detail\n")
print(noise_detail.to_string(index=False))

# ---- cell 11 ----
drop_rows = []
for label, R in RUNS.items():
    L, c = R["layer"], R["conds"]
    noise = R.get("noise_median", np.nan)
    for name, vals in c.items():
        k = kind_of(name)
        if k not in ("ablation", "random_control"):
            continue
        row = {"run": label, "condition": name, "kind": k,
               "dose": dose_of(name) if k == "ablation" else random_dose_of(name),
               "layer": L, "value@site": vals[L]}
        for dn, key in (("vs_SAE", "SAE_baseline"), ("vs_untouched", "no_SAE_baseline")):
            base = c.get(key)
            if base is None:
                row[f"baseline_{dn}"] = np.nan
                row[f"retained_{dn}"] = np.nan
                row[f"damage_{dn}"] = np.nan
                continue
            b = base[L]
            row[f"baseline_{dn}"] = b
            if abs(b) < MIN_BASELINE:
                row[f"retained_{dn}"] = np.nan
                row[f"damage_{dn}"] = np.nan
                flag("WARN", label,
                     f"'{name}': {key} at block {L} is {b:.4f}, below MIN_BASELINE: ratio suppressed")
            else:
                row[f"retained_{dn}"] = vals[L] / b
                row[f"damage_{dn}"] = 1 - vals[L] / b
        ref = c.get("SAE_baseline")
        raw_drop = abs(ref[L] - vals[L]) if ref else np.nan
        row["raw_drop"] = raw_drop
        row["drop_over_noise"] = raw_drop / noise if (noise and noise > 0) else np.nan
        row["above_noise"] = bool(np.isfinite(row["drop_over_noise"])
                                  and row["drop_over_noise"] >= NOISE_SIGMA_K)
        drop_rows.append(row)

drops = pd.DataFrame(drop_rows)

show = ["run", "condition", "dose", "baseline_vs_SAE", "value@site",
        "damage_vs_SAE", "damage_vs_untouched", "drop_over_noise", "above_noise"]
print("IMMEDIATE DROP AT THE INTERVENTION LAYER\n")
for label in RUNS:
    sub = drops[drops.run == label]
    if sub.empty:
        continue
    print(f"--- {label}  (block {RUNS[label]['layer']}) " + "-" * (46 - len(label)))
    print(sub[show].drop(columns=["run"]).to_string(index=False))
    print()

# ---- cell 12 ----
# agreement between the SAE-baseline and untouched-model denominators
agree_rows = []
for _, r in drops[drops.kind == "ablation"].iterrows():
    a, b = r["damage_vs_SAE"], r["damage_vs_untouched"]
    if not (np.isfinite(a) and np.isfinite(b)):
        continue
    agree_rows.append({"run": r["run"], "dose": int(r["dose"]),
                       "damage_vs_SAE": a, "damage_vs_untouched": b,
                       "abs_gap": abs(a - b),
                       "robust": abs(a - b) < 0.05})
agreement = pd.DataFrame(agree_rows)
print("Does the immediate-drop figure survive a change of denominator?\n")
print(agreement.to_string(index=False))
n_bad = int((~agreement.robust).sum())
if n_bad:
    flag("NOTE", "drop correction",
         f"{n_bad} dose(s) shift by >5 points between denominators: quote a range, not a point")
print(f"\n{len(agreement)-n_bad}/{len(agreement)} conditions agree to within 5 points.")

# ---- cell 14 ----
artifact_rows = []
for label, R in RUNS.items():
    c, L = R["conds"], R["layer"]
    sae, untouched = c.get("SAE_baseline"), c.get("no_SAE_baseline")
    if sae is None or untouched is None:
        flag("ERROR", label, "cannot compute artifact diagnostic: a baseline is missing")
        continue
    for b in range(N_BLOCKS):
        artifact_rows.append({
            "run": label, "block": b, "post_intervention": b >= L,
            "untouched": untouched[b], "SAE_baseline": sae[b],
            "difference": sae[b] - untouched[b],
        })
artifact = pd.DataFrame(artifact_rows)

for label in RUNS:
    sub = artifact[(artifact.run == label) & (artifact.post_intervention)]
    if sub.empty:
        continue
    infl = sub.difference.max()
    mean_infl = sub.difference.mean()
    print(f"--- {label} " + "-" * (50 - len(label)))
    print(sub[["block", "untouched", "SAE_baseline", "difference"]].to_string(index=False))
    print(f"    max inflation {infl:+.3f}   mean {mean_infl:+.3f}")
    if infl > 0.10:
        flag("WARN", label,
             f"SAE reconstruction inflates SSDC by up to {infl:+.3f} post-intervention: "
             "recovery measured against this baseline is not trustworthy")
    print()

# ---- cell 15 ----
n = len(RUNS)
fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), sharey=False)
axes = np.atleast_1d(axes)
for ax, (label, R) in zip(axes, RUNS.items()):
    sub = artifact[artifact.run == label]
    ax.plot(sub.block, sub.untouched, "o-", label="untouched model", lw=1.8, ms=4)
    ax.plot(sub.block, sub.SAE_baseline, "s--", label="SAE baseline", lw=1.8, ms=4)
    ax.axvline(R["layer"], color="k", ls=":", lw=1.2, alpha=0.7)
    ax.text(R["layer"], ax.get_ylim()[1], " intervention", fontsize=7,
            va="top", ha="left", rotation=90, alpha=0.7)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("block")
    ax.grid(alpha=0.25)
axes[0].set_ylabel("SSDC under RPI")
axes[0].legend(fontsize=8)
fig.suptitle("Reconstruction-artifact diagnostic: where the two baselines diverge", y=1.02)
fig.tight_layout()
plt.show()

# ---- cell 17 ----
def retention_curve(vals, base, L):
    out = []
    for b in range(L, N_BLOCKS):
        d = base[b]
        out.append(vals[b] / d if abs(d) >= MIN_BASELINE else np.nan)
    return np.array(out, dtype=float)

def half_recovery(ret):
    '''Blocks-after-intervention at which retention first reaches r0 + 0.5*(1 - r0).'''
    if len(ret) == 0 or not np.isfinite(ret[0]) or ret[0] >= 1.0:
        return None
    target = ret[0] + 0.5 * (1.0 - ret[0])
    for i in range(1, len(ret)):
        if not np.isfinite(ret[i]) or ret[i] < target:
            continue
        prev = ret[i - 1]
        if not np.isfinite(prev) or ret[i] == prev:
            return float(i)
        return float(i - 1 + (target - prev) / (ret[i] - prev))
    return None

records, curves = [], {}
for label, R in RUNS.items():
    L, c = R["layer"], R["conds"]
    noise = R.get("noise_median", np.nan)
    for name, vals in c.items():
        k = kind_of(name)
        if k not in ("ablation", "random_control"):
            continue
        for dn, key in (("vs_SAE", "SAE_baseline"), ("vs_untouched", "no_SAE_baseline")):
            base = c.get(key)
            if base is None:
                continue
            ret = retention_curve(vals, base, L)
            curves[(label, name, dn)] = ret

            r0 = ret[0] if len(ret) else np.nan
            damage = 1 - r0 if np.isfinite(r0) else np.nan
            raw_drop = abs(base[L] - vals[L])
            snr = raw_drop / noise if (noise and noise > 0) else np.nan

            reasons = []
            if not np.isfinite(r0):
                reasons.append("denominator below MIN_BASELINE")
            if np.isfinite(damage) and damage < DAMAGE_GATE:
                reasons.append(f"damage {damage:.1%} < {DAMAGE_GATE:.0%} gate")
            if np.isfinite(snr) and snr < NOISE_SIGMA_K:
                reasons.append(f"drop {snr:.1f}x noise < {NOISE_SIGMA_K}x")
            reportable = (len(reasons) == 0) and k == "ablation"

            hr = half_recovery(ret) if reportable else None
            final = ret[-1] if len(ret) and np.isfinite(ret[-1]) else np.nan
            overshoot = bool(np.isfinite(final) and final > OVERSHOOT_TOL)

            records.append({
                "run": label, "condition": name, "kind": k,
                "dose": dose_of(name) if k == "ablation" else random_dose_of(name),
                "denominator": dn,
                "damage@site": damage, "drop_over_noise": snr,
                "reportable": reportable,
                "suppressed_because": "; ".join(reasons) if reasons else "",
                "half_recovery_blocks": hr,
                "retention_final": final, "overshoot": overshoot,
                "nan_blocks": int(np.sum(~np.isfinite(ret))),
            })

master = pd.DataFrame(records)
print(f"{len(master)} condition x denominator rows computed.")
print(f"reportable: {int(master.reportable.sum())}   suppressed: {int((~master.reportable).sum())}")

# ---- cell 18 ----
print("=" * 100)
print("REPORTABLE : passed every guard")
print("=" * 100)
rep = master[master.reportable].sort_values(["run", "denominator", "dose"])
cols = ["run", "condition", "denominator", "damage@site", "drop_over_noise",
        "half_recovery_blocks", "retention_final", "overshoot"]
print(rep[cols].to_string(index=False) if len(rep) else "  (none)")

print()
print("=" * 100)
print("QUARANTINED: computed, but not reportable. Reason given for every row.")
print("=" * 100)
sup = master[~master.reportable].sort_values(["run", "denominator", "kind", "dose"])
cols_s = ["run", "condition", "kind", "denominator", "damage@site",
          "drop_over_noise", "suppressed_because"]
print(sup[cols_s].to_string(index=False) if len(sup) else "  (none)")

# ---- cell 19 ----
# dose-invariance of the repair rate
print("HALF-RECOVERY BY DOSE  (blocks; '-' = not reportable, 'never' = no recovery)\n")
inv_rows = []
for dn in ("vs_SAE", "vs_untouched"):
    print(f"--- denominator: {dn} " + "-" * 40)
    for label in RUNS:
        sub = master[(master.run == label) & (master.denominator == dn)
                     & (master.kind == "ablation")].sort_values("dose")
        if sub.empty:
            continue
        cells_txt, vals = [], []
        for _, r in sub.iterrows():
            if not r.reportable:
                cells_txt.append(f"{int(r['dose'])}:-")
            elif r["half_recovery_blocks"] is None or not np.isfinite(r["half_recovery_blocks"] or np.nan):
                cells_txt.append(f"{int(r['dose'])}:never")
            else:
                cells_txt.append(f"{int(r['dose'])}:{r['half_recovery_blocks']:.2f}")
                vals.append(r["half_recovery_blocks"])
        spread = (max(vals) - min(vals)) if len(vals) >= 2 else np.nan
        inv_rows.append({"run": label, "denominator": dn,
                         "n_measurable": len(vals),
                         "spread_blocks": spread,
                         "min": min(vals) if vals else np.nan,
                         "max": max(vals) if vals else np.nan})
        sp = f"{spread:.2f}" if np.isfinite(spread) else "n/a"
        print(f"  {label:20s} {'  '.join(cells_txt):48s}  spread={sp}")
    print()

dose_invariance = pd.DataFrame(inv_rows)
print("Spread of half-recovery across doses (small = dose-invariant repair)\n")
print(dose_invariance.to_string(index=False))

for _, r in dose_invariance.iterrows():
    if r["n_measurable"] < 2:
        flag("NOTE", r["run"],
             f"only {int(r['n_measurable'])} measurable dose(s) under {r['denominator']}: "
             "dose-invariance cannot be assessed")

# ---- cell 20 ----
# retention curves, one panel per run; suppressed conditions are drawn faint and dashed
DENOM = "vs_SAE"   # switch to "vs_untouched" and re-run to see the other normalisation

n = len(RUNS)
fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.9), sharey=True)
axes = np.atleast_1d(axes)
cmap = plt.get_cmap("viridis")

for ax, (label, R) in zip(axes, RUNS.items()):
    sub = master[(master.run == label) & (master.denominator == DENOM)
                 & (master.kind == "ablation")].sort_values("dose")
    doses = sub.dose.tolist()
    for i, (_, r) in enumerate(sub.iterrows()):
        ret = curves[(label, r["condition"], DENOM)]
        x = np.arange(len(ret))
        colr = cmap(i / max(len(doses) - 1, 1))
        if r["reportable"]:
            ax.plot(x, ret, "o-", color=colr, lw=1.9, ms=4, label=f"{int(r['dose'])} feat")
        else:
            ax.plot(x, ret, "x--", color=colr, lw=1.0, ms=4, alpha=0.35,
                    label=f"{int(r['dose'])} feat (suppressed)")
    rc = master[(master.run == label) & (master.denominator == DENOM)
                & (master.kind == "random_control")]
    for _, r in rc.iterrows():
        ret = curves[(label, r["condition"], DENOM)]
        ax.plot(np.arange(len(ret)), ret, color="crimson", ls="-.", lw=1.4,
                alpha=0.8, label="random control")
    ax.axhline(1.0, color="k", lw=0.8, alpha=0.5)
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.3)
    ax.set_title(f"{label}\n(ablated at block {R['layer']})", fontsize=9)
    ax.set_xlabel("blocks after intervention")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.5, loc="lower right")

axes[0].set_ylabel(f"retention  (ablated / baseline, {DENOM})")
axes[0].set_ylim(-0.25, 1.35)
fig.suptitle(f"Retention curves: solid = reportable, faint dashed = suppressed  [{DENOM}]", y=1.03)
fig.tight_layout()
plt.show()

# ---- cell 22 ----
if not FLAGS:
    print("No flags raised.")
else:
    fl = pd.DataFrame(FLAGS)
    fl["severity"] = pd.Categorical(fl.severity, ["ERROR", "WARN", "NOTE"], ordered=True)
    fl = fl.sort_values(["severity", "scope"]).reset_index(drop=True)
    for sev in ["ERROR", "WARN", "NOTE"]:
        sub = fl[fl.severity == sev]
        if sub.empty:
            continue
        print(f"\n{sev}  ({len(sub)})")
        print("-" * 92)
        for _, r in sub.iterrows():
            print(f"  [{r['scope']}] {r['message']}")
    print()
    print("=" * 92)
    print(f"totals: ERROR {int((fl.severity=='ERROR').sum())} | "
          f"WARN {int((fl.severity=='WARN').sum())} | NOTE {int((fl.severity=='NOTE').sum())}")

# ---- cell 23 ----
# Per-run verdict: is this run usable for a recovery claim at all?
print("PER-RUN VERDICT\n")
verdicts = []
for label, R in RUNS.items():
    sub = master[(master.run == label) & (master.kind == "ablation")]
    n_rep = int(sub.reportable.sum())
    max_dmg = np.nanmax(sub["damage@site"]) if len(sub) else np.nan
    max_snr = np.nanmax(sub["drop_over_noise"]) if len(sub) else np.nan
    if n_rep == 0:
        verdict = "NOT USABLE for recovery: no dose clears the guards"
    elif n_rep < 2:
        verdict = "WEAK: one measurable dose; no dose-invariance claim possible"
    else:
        verdict = f"USABLE: {n_rep} measurable condition-rows"
    verdicts.append({"run": label, "max_damage@site": max_dmg,
                     "max_drop_over_noise": max_snr,
                     "reportable_rows": n_rep, "verdict": verdict})
    print(f"  {label:20s} max damage {max_dmg:6.1%}   max SNR {max_snr:6.1f}x   {verdict}")

verdict_df = pd.DataFrame(verdicts)

# ---- cell 25 ----
OUT_DIR = "w1a_output"
os.makedirs(OUT_DIR, exist_ok=True)

def clean(df):
    return json.loads(df.replace({np.nan: None}).to_json(orient="records"))

export = {
    "notebook": "W1-A",
    "generated_utc": RUN_STAMP,
    "config": {"N_BLOCKS": N_BLOCKS, "DAMAGE_GATE": DAMAGE_GATE,
               "MIN_BASELINE": MIN_BASELINE, "NOISE_SIGMA_K": NOISE_SIGMA_K,
               "OVERSHOOT_TOL": OVERSHOOT_TOL},
    "runs": {label: {"seed": R["seed"], "family": R["family"],
                     "intervention_layer": R["layer"],
                     "source_file": R["source_file"],
                     "noise_median": R.get("noise_median"),
                     "noise_max": R.get("noise_max")}
             for label, R in RUNS.items()},
    "inventory": clean(inventory),
    "noise_detail": clean(noise_detail),
    "immediate_drop": clean(drops),
    "denominator_agreement": clean(agreement),
    "artifact_diagnostic": clean(artifact),
    "retention_master": clean(master),
    "dose_invariance": clean(dose_invariance),
    "verdicts": clean(verdict_df),
    "flags": FLAGS,
    "normalisation_protocol": {
        "formula": "(observed - floor) / (baseline - floor)",
        "floor_source": "W1-B (GPU): not yet available",
        "floors": {label: None for label in RUNS},
    },
    "retention_curves": {
        f"{label}||{cond}||{dn}": [None if not np.isfinite(v) else float(v) for v in arr]
        for (label, cond, dn), arr in curves.items()
    },
}

json_path = os.path.join(OUT_DIR, f"w1a_results_{RUN_STAMP.replace(':','-')}.json")
with open(json_path, "w") as fh:
    json.dump(export, fh, indent=2)

csvs = {"immediate_drop": drops, "retention_master": master,
        "artifact_diagnostic": artifact, "noise_detail": noise_detail,
        "dose_invariance": dose_invariance, "verdicts": verdict_df}
csv_paths = []
for name, df in csvs.items():
    p = os.path.join(OUT_DIR, f"{name}.csv")
    df.to_csv(p, index=False)
    csv_paths.append(p)

print("written:")
for p in [json_path] + csv_paths:
    print(f"  {p}  ({os.path.getsize(p):,} bytes)")

# ---- cell 26 ----
# Download everything as one archive (Colab only).
import shutil
archive = shutil.make_archive("w1a_output", "zip", OUT_DIR)
print(f"archive: {archive} ({os.path.getsize(archive):,} bytes)")

if IN_COLAB:
    from google.colab import files as colab_files
    colab_files.download(archive)
else:
    print("Not in Colab: collect the files from", os.path.abspath(OUT_DIR))

# ---- cell 28 ----
print("W1-A SUMMARY".center(92, "="))
print()

# 1. immediate drop
for label in RUNS:
    sub = drops[(drops.run == label) & (drops.kind == "ablation")]
    if sub.empty:
        continue
    top = sub.loc[sub["damage_vs_SAE"].idxmax()]
    print(f"[drop] {label}: at block {int(top['layer'])}, {int(top['dose'])} features take "
          f"SSDC from {top['baseline_vs_SAE']:.3f} to {top['value@site']:.3f} "
          f"({top['damage_vs_SAE']:.0%} removed vs SAE baseline, "
          f"{top['damage_vs_untouched']:.0%} vs untouched).")
print()

# 2. artifact
for label in RUNS:
    sub = artifact[(artifact.run == label) & artifact.post_intervention]
    if sub.empty:
        continue
    print(f"[artifact] {label}: reconstruction shifts SSDC by "
          f"{sub.difference.min():+.3f} to {sub.difference.max():+.3f} after the "
          f"intervention layer (mean {sub.difference.mean():+.3f}).")
print()

# 3. dose invariance
for _, r in dose_invariance[dose_invariance.denominator == "vs_untouched"].iterrows():
    if r["n_measurable"] >= 2:
        print(f"[repair] {r['run']}: half-recovery spans {r['min']:.2f}-{r['max']:.2f} blocks "
              f"across {int(r['n_measurable'])} doses (spread {r['spread_blocks']:.2f}).")
    else:
        print(f"[repair] {r['run']}: not enough measurable doses to state a repair rate.")
print()

for _, r in verdict_df.iterrows():
    print(f"[verdict] {r['run']}: {r['verdict']}")
print()
print("=" * 92)
print("Floors from W1-B are still required before any cross-model effect size is quoted.")

# ---- cell 29 ----

