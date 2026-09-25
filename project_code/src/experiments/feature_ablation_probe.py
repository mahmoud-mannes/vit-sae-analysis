"""SAE feature ablation: impact on linear position probes.

Same experiment as feature_ablation_ssdc.py (same conditions, same SAE / feature
selection / paths, see that file's docstring), but the readout is how well token
position can still be decoded from the activations after the ablation.

Two probes, chosen with probe_mode:

  "classification"  the normal probe: one linear layer that predicts each token's
                    exact position index (num_tokens classes, class token included).
                    Trained with metrics.position_probe.train_probe_memmap, the
                    protocol in docs/LINEAR_PROBE.md (lr 5e-3, 20 passes, batch 512,
                    last 10% of the images held out). Reported: peak held-out top-1
                    accuracy over the passes (chance = 1 / num_tokens).

  "regression"      a ridge probe that predicts the normalized row and the column of
                    every patch token (metrics.position_probe.evaluate_coordinate_probe).
                    Train and test are different images and different RPI permutations,
                    alpha is picked on a validation split, and a shuffled-label
                    control is fitted alongside. Reported: mean test R^2 of row and
                    column (a control R^2 near 0 means the probe is not fitting noise).

Both run under RPI by default (position can then only come from the positional
signal, not from image content).

Where the probe reads
---------------------
`probe_layers` (default: only the ablation layer, like the tables in
docs/LINEAR_PROBE.md) says which blocks are probed; "downstream" probes every
block from the ablation layer to the last, "all" probes every block, or pass a
list of indices. Each probed layer costs one forward pass and one probe fit. The
condition score is the peak over the probed layers. `probe_block` (default: the
ablated block) says which stream is read: "residual" is the residual entering
the block, "attention"/"mlp" are the sublayer outputs.

The ablation hook stays attached while the activations are collected, so the probe
sees the ablated model.

Run
---
    python feature_ablation_probe.py --probe-mode classification --sae-dir /path/to/SAE_Models
    python feature_ablation_probe.py --probe-mode regression --model-type RoPE --sae-dir ... --plot
    python feature_ablation_probe.py --probe-layers downstream --k-values 4 8 20 --sae-dir ...

From a notebook:
    from experiments.feature_ablation_probe import run_all
    out = run_all(model_type="APE", seed=1, sae_dir="...", probe_mode="regression", do_plot=True)
"""

import argparse
import gc
import os
import sys
import time

# Make this file work as a script, as `python -m`, and as an import.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.abspath(os.path.join(_HERE, ".."))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

import common
from main.load_models import get_block_attention, get_block_mlp, get_vit_blocks
from metrics.position_probe import (
    DEFAULT_COORDINATE_PROBE_ALPHA_GRID,
    evaluate_coordinate_probe,
    split_image_indices,
    train_probe_memmap,
)
from feature_ablation_ssdc import (
    BLOCKS,
    ablation_context,
    add_shared_cli_args,
    install_seeded_rpi_hook,
    results_path,
    run_forward,
    setup_experiment,
    shared_kwargs_from_args,
    write_results,
)

PROBE_MODES = ("classification", "regression")


# --------------------------------------------------------------------------- #
# Collecting activations
# --------------------------------------------------------------------------- #

def resolve_probe_layers(probe_layers, layer, n_layers):
    """None -> [layer]; "downstream" -> layer..last; "all" -> every block; else a list of indices."""
    if probe_layers is None:
        return [layer]
    if isinstance(probe_layers, str):
        if probe_layers == "downstream":
            return list(range(layer, n_layers))
        if probe_layers == "all":
            return list(range(n_layers))
        raise ValueError(f"probe_layers must be None, 'downstream', 'all' or a list, got {probe_layers!r}")
    return common.validate_layer_indices(probe_layers, n_layers, name="probe_layers")


def capture_activations(
    model, source, batches, layer, block, n_prefix, RPI=True, rpi_seed=0, drop_prefix=False
):
    """Collect one site's activations for every cached image as a CPU float32 tensor.

    block "residual": the residual stream entering block `layer` (as seen after any
    hook that changes the block input, so an ablation at this layer is included).
    block "attention" / "mlp": the output of that sublayer.
    Returns [num_images, num_tokens, d_model]; drop_prefix removes the class (and
    register) tokens so that only the patch grid is left.
    """
    blocks = get_vit_blocks(model, source)
    if block == "residual":
        target, from_input = blocks[layer], True
    elif block == "attention":
        target, from_input = get_block_attention(blocks[layer], source), False
    elif block == "mlp":
        target, from_input = get_block_mlp(blocks[layer], source)[0], False
    else:
        raise ValueError(f"block must be one of {BLOCKS}, got {block!r}")

    chunks = []

    def hook(module, inputs, output):
        tokens = inputs[0] if from_input else common.as_tensor(output)
        tokens = tokens.detach()
        if drop_prefix:
            tokens = tokens[:, n_prefix:, :]
        chunks.append(tokens.float().cpu())

    handles = [target.register_forward_hook(hook)]
    if RPI:
        handles.append(install_seeded_rpi_hook(model, source, rpi_seed))
    try:
        run_forward(model, source, batches)
    finally:
        for handle in handles:
            handle.remove()
    return torch.cat(chunks, dim=0)


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #

def fit_classification_probe(acts, lr, passes, batch_size, weight_decay, val_fraction, seed, device):
    """Linear position-index probe (docs/LINEAR_PROBE.md protocol). acts: [N, T, D] tensor."""
    torch.manual_seed(int(seed))
    _probe, history = train_probe_memmap(
        acts,
        probe_type="linear",
        num_passes=passes,
        lr=lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
        val_fraction=val_fraction,
        device=device,
    )
    return {
        "peak_accuracy": float(max(history["accuracy"])),
        "final_accuracy": float(history["accuracy"][-1]),
        "history": {key: [float(x) for x in values] for key, values in history.items()},
        "chance": 1.0 / int(acts.shape[1]),
    }


def fit_regression_probe(train_acts, eval_acts, split_seed, shuffle_seed, alpha_grid):
    """Ridge row/column probe. Both arrays: numpy [N, patch_tokens, D]."""
    split = split_image_indices(int(train_acts.shape[0]), seed=split_seed)
    return evaluate_coordinate_probe(
        train_acts, eval_acts, split, shuffle_seed=shuffle_seed, alpha_grid=alpha_grid
    )


def measure_layer(exp, probe_layer, probe_block, cfg):
    """Probe one layer under the currently attached ablation. Returns (score, detail)."""
    if cfg["mode"] == "classification":
        acts = capture_activations(
            exp.model, exp.source, exp.batches, probe_layer, probe_block, exp.n_prefix,
            RPI=cfg["RPI"], rpi_seed=cfg["rpi_seed"], drop_prefix=False,
        )
        detail = fit_classification_probe(
            acts, cfg["lr"], cfg["passes"], cfg["batch_size"], cfg["weight_decay"],
            cfg["val_fraction"], cfg["probe_seed"], cfg["device"],
        )
        detail = {"layer": probe_layer, **detail}
        return detail["peak_accuracy"], detail

    train_acts = capture_activations(
        exp.model, exp.source, exp.batches, probe_layer, probe_block, exp.n_prefix,
        RPI=cfg["RPI"], rpi_seed=cfg["rpi_seed"], drop_prefix=True,
    ).numpy()
    if cfg["RPI"]:
        # Test on a different permutation than the one the probe was trained on.
        eval_acts = capture_activations(
            exp.model, exp.source, exp.batches, probe_layer, probe_block, exp.n_prefix,
            RPI=True, rpi_seed=cfg["rpi_seed"] + 1, drop_prefix=True,
        ).numpy()
    else:
        eval_acts = train_acts
    detail = fit_regression_probe(
        train_acts, eval_acts, cfg["split_seed"],
        shuffle_seed=cfg["rpi_seed"] + 1009 * (probe_layer + 1), alpha_grid=cfg["alpha_grid"],
    )
    detail = {"layer": probe_layer, **detail}
    return detail["mean_test_r2"], detail


def summarize_probe(scores, layers):
    scores = np.asarray(scores, dtype=float)
    best = int(np.argmax(scores))
    return {
        "peak": float(scores[best]),
        "peak_layer": int(layers[best]),
        "final": float(scores[-1]),
        "mean": float(scores.mean()),
    }


# --------------------------------------------------------------------------- #
# Experiment orchestration
# --------------------------------------------------------------------------- #

def run_all(
    dataset=None,
    model_type="APE",
    seed=1,
    layer=None,
    block="residual",
    # SAE
    sae_dir=None,
    sae_filename=None,
    sae_topk=64,
    d_model=768,
    d_multiplier=5,
    architecture="topk",
    # features
    selectivity_scores_path=None,
    axis="both",
    k_values=None,
    random_k_values=None,
    random_draws=1,
    exclude_positional_from_random=False,
    condition_seed=0,
    # data
    number_images=1000,
    batch_size=128,
    RPI=True,
    rpi_seed=0,
    shuffle_dataset=True,
    data_seed=0,
    hf_token=None,
    # probe
    probe_mode="classification",
    probe_layers=None,
    probe_block=None,
    probe_seed=0,
    probe_lr=5e-3,
    probe_passes=20,
    probe_batch_size=512,
    probe_weight_decay=1e-4,
    probe_val_fraction=0.1,
    alpha_grid=DEFAULT_COORDINATE_PROBE_ALPHA_GRID,
    split_seed=0,
    # outputs
    output_dir=None,
    tag="",
    save=True,
    do_plot=False,
):
    """Run every condition, probe it, and return {metadata, conditions, curves, summary, details}.

    The arguments up to `hf_token` are the same as feature_ablation_ssdc.run_all
    (shuffle_dataset defaults to True here, as in the linear-probe notebook).

    probe_mode      "classification" (exact position, linear) or "regression" (row/column, ridge).
    probe_layers    None -> the ablation layer only; "downstream"; "all"; or a list of block indices.
    probe_block     stream to read: "residual" | "attention" | "mlp"; None -> the ablated block.
    probe_*         classification probe training settings (defaults: docs/LINEAR_PROBE.md).
    alpha_grid, split_seed   regression probe: ridge alphas and the image split seed.
    tag             appended to the results file name, to keep runs with different probe settings apart.

    out["curves"][condition] is the score at each probed layer (peak accuracy, or mean test R^2);
    out["summary"][condition]["peak"] is the number to compare across conditions.
    """
    if probe_mode not in PROBE_MODES:
        raise ValueError(f"probe_mode must be one of {PROBE_MODES}, got {probe_mode!r}")
    probe_block = probe_block or block
    if probe_block not in BLOCKS:
        raise ValueError(f"probe_block must be one of {BLOCKS}, got {probe_block!r}")

    exp = setup_experiment(
        model_type=model_type, seed=seed, layer=layer, block=block,
        sae_dir=sae_dir, sae_filename=sae_filename, sae_topk=sae_topk, d_model=d_model,
        d_multiplier=d_multiplier, architecture=architecture,
        selectivity_scores_path=selectivity_scores_path, axis=axis, k_values=k_values,
        random_k_values=random_k_values, random_draws=random_draws,
        exclude_positional_from_random=exclude_positional_from_random,
        condition_seed=condition_seed, dataset=dataset, hf_token=hf_token,
        shuffle_dataset=shuffle_dataset, data_seed=data_seed,
        number_images=number_images, batch_size=batch_size,
    )
    layers = resolve_probe_layers(probe_layers, exp.layer, exp.n_layers)
    cfg = {
        "mode": probe_mode, "RPI": bool(RPI), "rpi_seed": int(rpi_seed),
        "lr": probe_lr, "passes": probe_passes, "batch_size": probe_batch_size,
        "weight_decay": probe_weight_decay, "val_fraction": probe_val_fraction,
        "probe_seed": probe_seed, "split_seed": split_seed,
        "alpha_grid": tuple(float(a) for a in alpha_grid),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    print(
        f"{exp.model_type} seed {exp.seed} | ablate {exp.block} @ layer {exp.layer} | "
        f"{probe_mode} probe on {probe_block}, layers {layers} | {exp.number_images} images | "
        f"{len(exp.conditions)} conditions"
    )

    curves, details = {}, {}
    for condition in exp.conditions:
        print(f"-- condition: {condition.name} --")
        start = time.time()
        scores, layer_details = [], []
        with ablation_context(
            exp.model, exp.source, exp.model_type, exp.sae, exp.layer, exp.block, condition
        ):
            for probe_layer in layers:
                score, detail = measure_layer(exp, probe_layer, probe_block, cfg)
                scores.append(float(score))
                layer_details.append(detail)
                print(f"    layer {probe_layer:2d}: {score:.4f}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        curves[condition.name] = scores
        details[condition.name] = layer_details
        print(f"    ({time.time() - start:.1f}s)")

    metric = "peak_accuracy" if probe_mode == "classification" else "mean_test_r2"
    out = {
        "experiment": "feature_ablation_probe",
        **exp.metadata(),
        "RPI": bool(RPI),
        "rpi_seed": int(rpi_seed),
        "probe_mode": probe_mode,
        "probe_block": probe_block,
        "probe_layers": layers,
        "metric": metric,
        "probe_settings": (
            {"lr": probe_lr, "passes": probe_passes, "batch_size": probe_batch_size,
             "weight_decay": probe_weight_decay, "val_fraction": probe_val_fraction,
             "probe_seed": probe_seed}
            if probe_mode == "classification"
            else {"alpha_grid": list(cfg["alpha_grid"]), "split_seed": split_seed}
        ),
        "conditions": {c.name: c.spec() for c in exp.conditions},
        "curves": curves,
        "summary": {name: summarize_probe(curve, layers) for name, curve in curves.items()},
        "details": details,
    }
    print_summary(out)

    prefix = f"feature_ablation_probe_{probe_mode}"
    if save:
        write_results(
            out, results_path(prefix, exp.model_type, exp.seed, exp.block, exp.layer, output_dir, tag=tag)
        )
    if do_plot:
        figure = results_path(
            prefix, exp.model_type, exp.seed, exp.block, exp.layer,
            output_dir=common.FIGURES_DIR, ext=".png", tag=tag,
        )
        make_plots(out, save_path=figure)
    return out


def print_summary(out):
    label = "top-1 accuracy" if out["probe_mode"] == "classification" else "mean test R^2"
    print(f"\n==== {out['probe_mode']} probe, peak {label} over layers {out['probe_layers']} ====")
    header = f"{'condition':32s} {'peak':>8s} {'peak_L':>7s}"
    print(header)
    print("-" * len(header))
    for name, summary in out["summary"].items():
        print(f"{name:32s} {summary['peak']:8.4f} {summary['peak_layer']:7d}")


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def make_plots(out, save_path=None):
    """Dose-response: probe score against the number of ablated features."""
    import matplotlib.pyplot as plt

    conditions, summary = out["conditions"], out["summary"]
    classification = out["probe_mode"] == "classification"

    fig, ax = plt.subplots(figsize=(7, 4.5))

    top = sorted((conditions[n]["k"], summary[n]["peak"]) for n in conditions if conditions[n]["kind"] == "top")
    if top:
        ax.plot([k for k, _ in top], [v for _, v in top], marker="o", color="tab:blue",
                label="top positional features")

    random_by_k = {}
    for name, spec in conditions.items():
        if spec["kind"] == "random":
            random_by_k.setdefault(spec["k"], []).append(summary[name]["peak"])
    if random_by_k:
        ks = sorted(random_by_k)
        ax.plot(ks, [float(np.mean(random_by_k[k])) for k in ks], marker="s", linestyle="--",
                color="tab:gray", label="random features (mean)")
        if any(len(v) > 1 for v in random_by_k.values()):
            ax.scatter([k for k in ks for _ in random_by_k[k]],
                       [v for k in ks for v in random_by_k[k]], color="tab:gray", alpha=0.35, s=14)

    for name, style, label in (("no_SAE_baseline", "-", "no SAE"), ("SAE_baseline", "--", "SAE reconstruction")):
        if name in summary:
            ax.axhline(summary[name]["peak"], color="black", linestyle=style, linewidth=1.2, label=label)
    if classification:
        chance = out["details"]["no_SAE_baseline"][0]["chance"]
        ax.axhline(chance, color="tab:red", linestyle=":", linewidth=1.2, label="chance")
    else:
        ax.axhline(0.0, color="0.7", linewidth=0.8, zorder=0)

    layers = out["probe_layers"]
    where = f"layer {layers[0]}" if len(layers) == 1 else f"peak over layers {layers[0]}-{layers[-1]}"
    ax.set_xlabel("features ablated (k)")
    ax.set_ylabel("top-1 accuracy" if classification else "mean test R$^2$ (row, column)")
    ax.set_title(f"{out['model_type']} seed {out['seed']}: {out['probe_mode']} probe, {where}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(str(save_path)), exist_ok=True)
        fig.savefig(str(save_path), dpi=150)
        print(f"saved {save_path}")
    return fig


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #

def _parse_probe_layers(values):
    if not values:
        return None
    if len(values) == 1 and values[0] in ("downstream", "all"):
        return values[0]
    return [int(v) for v in values]


def main(argv=None):
    parser = argparse.ArgumentParser(description="SAE feature ablation: impact on linear position probes.")
    add_shared_cli_args(parser)
    parser.add_argument("--probe-mode", choices=PROBE_MODES, default="classification")
    parser.add_argument("--probe-layers", nargs="*", default=None,
                        help="block indices, or 'downstream' / 'all'; default: the ablation layer only")
    parser.add_argument("--probe-block", choices=BLOCKS, default=None,
                        help="stream to probe; default: the ablated block")
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--probe-lr", type=float, default=5e-3)
    parser.add_argument("--probe-passes", type=int, default=20)
    parser.add_argument("--probe-batch-size", type=int, default=512)
    parser.add_argument("--alpha-grid", type=float, nargs="+", default=list(DEFAULT_COORDINATE_PROBE_ALPHA_GRID))
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--no-shuffle-dataset", action="store_true")
    args = parser.parse_args(argv)
    return run_all(
        **shared_kwargs_from_args(args),
        shuffle_dataset=not args.no_shuffle_dataset,
        probe_mode=args.probe_mode,
        probe_layers=_parse_probe_layers(args.probe_layers),
        probe_block=args.probe_block,
        probe_seed=args.probe_seed,
        probe_lr=args.probe_lr,
        probe_passes=args.probe_passes,
        probe_batch_size=args.probe_batch_size,
        alpha_grid=tuple(args.alpha_grid),
        split_seed=args.split_seed,
    )


if __name__ == "__main__":
    main()
