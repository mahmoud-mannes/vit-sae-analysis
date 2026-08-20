"""Regenerate the curated causal-follow-up figures from committed JSON runs."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = REPO_ROOT / "results" / "runs" / "imagenet1k_val" / "causal_followups"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "figures"

COLORS = {
    "ape": "#2878B5",
    "rope": "#D45A4C",
    "attention": "#2878B5",
    "mlp": "#E07A35",
    "residual": "#4C78A8",
}


def load_json(name):
    with (RUNS_DIR / name).open() as handle:
        return json.load(handle)


def load_model_pair(prefix, model):
    return [load_json(f"{prefix}_{model}_seed{seed}.json") for seed in (0, 1)]


def mean_and_range(values):
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=0)
    error = np.vstack((mean - values.min(axis=0), values.max(axis=0) - mean))
    return mean, error


def style_axis(axis, zero=True):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
    axis.set_axisbelow(True)
    if zero:
        axis.axhline(0, color="#555555", linewidth=0.8)


def save(figure, output_dir, name):
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_dir / f"{name}.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def normalized_recovery(run, condition, metric, layer=None):
    if layer is None:
        clean = run["baselines"]["clean"][metric]
        rpi = run["baselines"]["rpi"][metric]
        patched = run["patches"][condition][metric]
    else:
        clean = run["baselines"]["clean"]["scores"][layer]
        rpi = run["baselines"]["rpi"]["scores"][layer]
        patched = run["patches"][condition]["scores"][layer]
    denominator = clean - rpi
    return 0.0 if abs(denominator) < 1e-12 else (patched - rpi) / denominator


def plot_final_layer_baselines(output_dir):
    runs = [
        load_model_pair("final_layer_activation_patching", model)[0]
        for model in ("ape", "rope")
    ]
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    x = np.arange(2)
    width = 0.35
    for axis, metric, title in zip(
        axes,
        ("accuracy", "final"),
        ("Top-1 accuracy", "Final-layer SSDC"),
    ):
        clean = [run["baselines"]["clean"][metric] for run in runs]
        rpi = [run["baselines"]["rpi"][metric] for run in runs]
        axis.bar(x - width / 2, clean, width, label="Normal order", color="#2A9D8F")
        axis.bar(x + width / 2, rpi, width, label="RPI", color="#E76F51")
        axis.set_xticks(x, ("APE", "RoPE"))
        axis.set_title(title)
        axis.set_ylabel("Score")
        axis.set_ylim(0, max(clean + rpi) * 1.2)
        style_axis(axis, zero=False)
    axes[0].legend(frameon=False, loc="upper right")
    save(figure, output_dir, "causal_final_layer_baselines")


def plot_final_layer_patching(output_dir):
    figure, axes = plt.subplots(2, 2, figsize=(8.8, 6.4), sharex=True)
    layers = np.arange(12)
    components = (
        ("residual", "Residual", COLORS["residual"]),
        ("attn", "Attention", COLORS["attention"]),
        ("mlp", "MLP", COLORS["mlp"]),
    )
    for row, model in enumerate(("ape", "rope")):
        runs = load_model_pair("final_layer_activation_patching", model)
        for column, (metric, label) in enumerate(
            (("final", "Final-layer SSDC recovery"), ("accuracy", "Accuracy recovery"))
        ):
            axis = axes[row, column]
            for component, component_label, color in components:
                values = [
                    [
                        normalized_recovery(run, f"{component}_L{layer:02d}", metric)
                        for layer in layers
                    ]
                    for run in runs
                ]
                mean, error = mean_and_range(values)
                axis.plot(layers, mean, marker="o", markersize=3, color=color, label=component_label)
                axis.fill_between(layers, mean - error[0], mean + error[1], color=color, alpha=0.12)
            axis.axhline(1, color="#666666", linestyle="--", linewidth=0.9)
            axis.set_title(f"{model.upper()}: {label}")
            axis.set_ylabel("Normalized recovery")
            style_axis(axis)
    for axis in axes[-1]:
        axis.set_xlabel("Patched layer")
        axis.set_xticks(layers)
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8)
    save(figure, output_dir, "causal_final_layer_patching")


def final_ablation_loss(run, condition, order, metric):
    return run["baselines"][order][metric] - run["ablations"][condition][order][metric]


def plot_final_layer_ablation(output_dir):
    figure, axes = plt.subplots(3, 2, figsize=(8.8, 7.2), sharex=True)
    layers = np.arange(12)
    readouts = (
        ("clean", "final", "Normal-order final SSDC loss"),
        ("rpi", "final", "RPI final SSDC loss"),
        ("clean", "accuracy", "Normal-order accuracy loss"),
    )
    for column, model in enumerate(("ape", "rope")):
        runs = load_model_pair("final_layer_zero_ablation", model)
        for row, (order, metric, label) in enumerate(readouts):
            axis = axes[row, column]
            for component, color in (("attn", COLORS["attention"]), ("mlp", COLORS["mlp"])):
                values = [
                    [
                        final_ablation_loss(run, f"{component}_L{layer:02d}", order, metric)
                        for layer in layers
                    ]
                    for run in runs
                ]
                mean, error = mean_and_range(values)
                axis.plot(layers, mean, marker="o", markersize=3, color=color, label=component.title())
                axis.fill_between(layers, mean - error[0], mean + error[1], color=color, alpha=0.12)
            axis.set_title(f"{model.upper()}: {label}")
            axis.set_ylabel("Baseline - ablated")
            style_axis(axis)
    for axis in axes[-1]:
        axis.set_xlabel("Ablated layer")
        axis.set_xticks(layers)
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    save(figure, output_dir, "causal_final_layer_ablation")


def plot_peak_layer_patching(output_dir):
    specs = [f"attn_L{layer:02d}" for layer in range(6)] + [
        f"mlp_L{layer:02d}" for layer in range(6)
    ]
    labels = [f"Attn {layer}" for layer in range(6)] + [f"MLP {layer}" for layer in range(6)]
    colors = [COLORS["attention"]] * 6 + [COLORS["mlp"]] * 6
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 6.8), sharex=True)
    for row, model in enumerate(("ape", "rope")):
        runs = load_model_pair("peak_layer_activation_patching", model)
        for column, layer in enumerate((4, 5)):
            values = [
                [normalized_recovery(run, spec, "scores", layer=layer) for spec in specs]
                for run in runs
            ]
            mean, error = mean_and_range(values)
            axis = axes[row, column]
            axis.bar(np.arange(len(specs)), mean, color=colors, width=0.78)
            axis.errorbar(np.arange(len(specs)), mean, yerr=error, fmt="none", ecolor="#222222", capsize=2)
            axis.axhline(1, color="#666666", linestyle="--", linewidth=0.9)
            axis.set_title(f"{model.upper()}, SSDC read at layer {layer}")
            axis.set_ylabel("Normalized SSDC recovery")
            style_axis(axis)
    for axis in axes[-1]:
        axis.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    save(figure, output_dir, "causal_peak_layer_patching")


def peak_ablation_loss(run, spec, layer):
    key = str(layer)
    return run["baselines"]["rpi"]["ssdc_by_layer"][key] - run["ablations"][spec]["rpi"]["ssdc_by_layer"][key]


def plot_peak_layer_ablation(output_dir):
    specs = [f"attn_L{layer:02d}" for layer in range(6)] + [
        f"mlp_L{layer:02d}" for layer in range(6)
    ]
    labels = [f"Attn {layer}" for layer in range(6)] + [f"MLP {layer}" for layer in range(6)]
    colors = [COLORS["attention"]] * 6 + [COLORS["mlp"]] * 6
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 6.8), sharex=True)
    for row, model in enumerate(("ape", "rope")):
        runs = load_model_pair("peak_layer_zero_ablation", model)
        for column, layer in enumerate((4, 5)):
            values = [[peak_ablation_loss(run, spec, layer) for spec in specs] for run in runs]
            mean, error = mean_and_range(values)
            axis = axes[row, column]
            axis.bar(np.arange(len(specs)), mean, color=colors, width=0.78)
            axis.errorbar(np.arange(len(specs)), mean, yerr=error, fmt="none", ecolor="#222222", capsize=2)
            axis.set_title(f"{model.upper()}, SSDC read at layer {layer}")
            axis.set_ylabel("RPI SSDC loss after ablation")
            style_axis(axis)
    for axis in axes[-1]:
        axis.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    save(figure, output_dir, "causal_peak_layer_ablation")


def plot_position_alignment(output_dir):
    controls = ("same_image_aligned", "same_image_misaligned", "different_image_deranged", "mean_donor")
    labels = ("Same image\naligned", "Same image\nshifted", "Other image\naligned", "Mean donor")
    colors = ("#2D7D46", "#C84B31", "#4C78A8", "#8A8A8A")
    settings = (("ape", "mlp_L00", 4), ("rope", "attn_L05", 5))
    figure, axes = plt.subplots(2, 2, figsize=(9.4, 6.7))
    for row, (model, spec, layer) in enumerate(settings):
        runs = load_model_pair("position_alignment", model)
        for column, metric in enumerate(("ssdc", "accuracy")):
            values = []
            for run in runs:
                values.append([
                    run["controls"][spec][control]["accuracy"]
                    if metric == "accuracy"
                    else run["controls"][spec][control]["readout_layers"][str(layer)]
                    for control in controls
                ])
            mean, error = mean_and_range(values)
            axis = axes[row, column]
            axis.bar(np.arange(4), mean, color=colors, width=0.72)
            axis.errorbar(np.arange(4), mean, yerr=error, fmt="none", ecolor="#222222", capsize=2)
            axis.set_xticks(np.arange(4), labels)
            axis.set_ylabel(f"SSDC at layer {layer}" if metric == "ssdc" else "Top-1 accuracy")
            axis.set_title(f"{model.upper()}: {'spatial structure' if metric == 'ssdc' else 'classification'}")
            if metric == "ssdc":
                rpi_baseline = np.mean(
                    [run["baselines"]["rpi"]["readout_layers"][str(layer)] for run in runs]
                )
                axis.axhline(
                    rpi_baseline,
                    color="#666666",
                    linestyle="--",
                    linewidth=0.9,
                    label="Unpatched RPI",
                )
                axis.legend(frameon=False, fontsize=7, loc="lower right")
            style_axis(axis, zero=False)
    save(figure, output_dir, "causal_position_alignment")


def attention_curve(runs, model, field):
    return np.asarray(
        [[layer[field]["rpi"] for layer in run["models"][model]["layers"]] for run in runs],
        dtype=float,
    )


def plot_seed_curve(axis, values, label, color, linestyle="-"):
    mean, error = mean_and_range(values)
    layers = np.arange(values.shape[1])
    axis.plot(layers, mean, color=color, linewidth=2, linestyle=linestyle, label=label)
    axis.fill_between(layers, mean - error[0], mean + error[1], color=color, alpha=0.13)


def attention_runs():
    return [load_json(f"attention_output_and_probes_seed{seed}.json") for seed in (0, 1)]


def plot_attention_outputs(output_dir):
    runs = attention_runs()
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 6.7), sharex=True)
    for row, model in enumerate(("ape", "rope")):
        raw_axis, delta_axis = axes[row]
        plot_seed_curve(
            raw_axis,
            attention_curve(runs, model, "attention_output_ssdc"),
            "Attention output",
            COLORS["attention"],
        )
        plot_seed_curve(raw_axis, attention_curve(runs, model, "mlp_output_ssdc"), "MLP output", COLORS["mlp"])
        plot_seed_curve(
            delta_axis,
            attention_curve(runs, model, "attention_residual_delta"),
            "Attention update",
            COLORS["attention"],
        )
        plot_seed_curve(delta_axis, attention_curve(runs, model, "mlp_residual_delta"), "MLP update", COLORS["mlp"])
        raw_axis.set_title(f"{model.upper()}: component-output SSDC")
        delta_axis.set_title(f"{model.upper()}: change in residual SSDC")
        raw_axis.set_ylabel("RPI SSDC")
        delta_axis.set_ylabel("SSDC after stage - before stage")
        for axis in (raw_axis, delta_axis):
            axis.set_xticks(range(12))
            axis.legend(frameon=False, fontsize=8)
            style_axis(axis)
    save(figure, output_dir, "causal_attention_output_ssdc")


def probe_curve(runs, model, field):
    return np.asarray(
        [[layer["probe_rpi"][field] for layer in run["models"][model]["layers"]] for run in runs],
        dtype=float,
    )


def plot_attention_probes(output_dir):
    runs = attention_runs()
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    for model in ("ape", "rope"):
        plot_seed_curve(axis, probe_curve(runs, model, "mean_test_r2"), model.upper(), COLORS[model])
        plot_seed_curve(
            axis,
            probe_curve(runs, model, "negative_control_mean_test_r2"),
            f"{model.upper()} shuffled labels",
            COLORS[model],
            linestyle="--",
        )
    axis.set_xticks(range(12))
    axis.set_xlabel("Layer")
    axis.set_ylabel("Held-out mean row/column $R^2$")
    axis.legend(frameon=False, ncol=2, fontsize=8)
    style_axis(axis)
    save(figure, output_dir, "causal_attention_position_probes")




def generate_all(output_dir):
    plot_final_layer_baselines(output_dir)
    plot_final_layer_patching(output_dir)
    plot_final_layer_ablation(output_dir)
    plot_peak_layer_patching(output_dir)
    plot_peak_layer_ablation(output_dir)
    plot_position_alignment(output_dir)
    plot_attention_outputs(output_dir)
    plot_attention_probes(output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    generate_all(args.output_dir)


if __name__ == "__main__":
    main()
