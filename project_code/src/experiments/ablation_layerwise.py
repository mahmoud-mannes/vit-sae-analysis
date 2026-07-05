"""Layer windowed MLP and attention ablation for the APE ViT.

Motivation
----------
Two working claims frame this experiment:

  (1) Ablating MLPs destroys SSDC recovery under RPI.
  (2) Ablating attention destroys the SSDC decay that happens in later layers.

Both are stated for whole model ablations. This script tightens them by ablating
a component only inside an early, middle, or late window, and by the reverse
"keep only" probe that leaves a component alive in just one window. The point is
to separate two explanations for the APE model's early SSDC peak:

  - "early layers are special": the early MLP blocks specifically build the index
    anchored structure, or
  - "first blocks encountered": any first surviving MLP blocks would, wherever
    they sit in the stack.

Concrete questions this answers:
  - Is the early SSDC-under-RPI peak removed by ablating only the early MLPs, and
    left intact by ablating only the late MLPs? (early MLP specificity)
  - If we keep MLPs alive only in the mid or late window, does the peak move to
    follow the first surviving MLPs? (first blocks encountered)
  - Does removing attention only in the later layers still flatten the later
    layer SSDC decay, or is the whole attention stack needed?

Primary metric is SSDC under RPI, since the early peak and the later decay both
live in that curve for the APE model. We also record clean SSDC for the baseline.

Run:
    python ablation_layerwise.py --number-images 512 --plot
    python ablation_layerwise.py --number-images 512 --plot --rank
"""

import argparse
import os
import sys

# Make this file work as a script, as `python -m`, and as an import.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.abspath(os.path.join(_HERE, ".."))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common
from interventions.ablation import AblationController, num_blocks
from main.load_models import load_model
from metrics.ssdc import evaluate_ssdc
from metrics.effective_rank import evaluate_effective_rank


def build_conditions(n_layers, early, mid, late):
    """Return an ordered dict-like list of (name, spec) where spec is either None
    (baseline) or (component, layers, mode)."""
    all_layers = list(range(n_layers))
    return [
        ("baseline", None),
        # (1) does MLP ablation destroy SSDC recovery, and is it early specific?
        ("mlp_zero_early", ("mlp", early, "zero")),
        ("mlp_zero_mid", ("mlp", mid, "zero")),
        ("mlp_zero_late", ("mlp", late, "zero")),
        ("mlp_zero_all", ("mlp", all_layers, "zero")),
        # first-blocks-encountered probe: keep MLPs alive in only one window
        ("mlp_keep_early", ("mlp", early, "keep_only")),
        ("mlp_keep_mid", ("mlp", mid, "keep_only")),
        ("mlp_keep_late", ("mlp", late, "keep_only")),
        # (2) does attention ablation destroy the later layer decay, late only?
        ("attn_zero_early", ("attn", early, "zero")),
        ("attn_zero_mid", ("attn", mid, "zero")),
        ("attn_zero_late", ("attn", late, "zero")),
        ("attn_zero_all", ("attn", all_layers, "zero")),
    ]


def run_condition(model, processor, dataset, source, spec, RPI, number_images, batch_size):
    kwargs = dict(RPI=RPI, number_images=number_images, batch_size=batch_size)
    if spec is None:
        scores, _ = evaluate_ssdc(model, processor, dataset, source, **kwargs)
        return scores
    component, layers, mode = spec
    with AblationController(model, source, component, layers, mode):
        scores, _ = evaluate_ssdc(model, processor, dataset, source, **kwargs)
    return scores


def run_rank_condition(model, processor, dataset, source, spec, RPI, number_images, batch_size):
    kwargs = dict(RPI=RPI, number_images=number_images, batch_size=batch_size)
    if spec is None:
        return evaluate_effective_rank(model, processor, dataset, source, **kwargs)
    component, layers, mode = spec
    with AblationController(model, source, component, layers, mode):
        return evaluate_effective_rank(model, processor, dataset, source, **kwargs)


def make_plots(rpi_curves, clean_baseline):
    mlp_zero = ["baseline", "mlp_zero_early", "mlp_zero_late", "mlp_zero_all"]
    mlp_keep = ["baseline", "mlp_keep_early", "mlp_keep_mid", "mlp_keep_late"]
    attn_zero = ["baseline", "attn_zero_early", "attn_zero_late", "attn_zero_all"]

    common.plot_curves(
        {k: rpi_curves[k] for k in mlp_zero if k in rpi_curves},
        title="APE SSDC under RPI: MLP ablation by window",
        ylabel="SSDC under RPI",
        save_path=os.path.join(common.FIGURES_DIR, "ablation_mlp_zero_rpi.png"),
    )
    common.plot_curves(
        {k: rpi_curves[k] for k in mlp_keep if k in rpi_curves},
        title="APE SSDC under RPI: keep MLPs in one window only",
        ylabel="SSDC under RPI",
        save_path=os.path.join(common.FIGURES_DIR, "ablation_mlp_keep_rpi.png"),
    )
    common.plot_curves(
        {k: rpi_curves[k] for k in attn_zero if k in rpi_curves},
        title="APE SSDC under RPI: attention ablation by window",
        ylabel="SSDC under RPI",
        save_path=os.path.join(common.FIGURES_DIR, "ablation_attn_zero_rpi.png"),
    )


def print_summary(rpi_curves):
    print("\n==== SSDC-under-RPI summary per condition ====")
    header = f"{'condition':18s} {'peak':>6s} {'peak_L':>7s} {'delta':>7s} {'decay':>7s} {'final':>7s} {'auc':>7s}"
    print(header)
    print("-" * len(header))
    for name, curve in rpi_curves.items():
        s = common.summarize_curve(curve)
        print(
            f"{name:18s} {s['peak']:6.3f} {s['peak_layer']:7d} {s['delta']:7.3f} "
            f"{s['decay']:7.3f} {s['final']:7.3f} {s['auc']:7.3f}"
        )
    print(
        "\nReading guide:\n"
        "  peak_L   where SSDC-under-RPI peaks. Baseline APE peaks early (blocks 2 to 3).\n"
        "  decay    peak minus final. A large baseline decay is the 'later layer decay'.\n"
        "  MLP claim:  mlp_zero_all should crush peak and auc toward zero.\n"
        "  early MLP:  if mlp_zero_early kills the early peak but mlp_zero_late does not,\n"
        "              the peak is tied to the early MLP blocks.\n"
        "  first-met:  if mlp_keep_mid / mlp_keep_late shift peak_L to follow the kept\n"
        "              window, the peak tracks the first surviving MLPs, not depth per se.\n"
        "  attn claim: attn_zero_all should shrink decay (SSDC stays high). If\n"
        "              attn_zero_late alone also shrinks decay, the decay is a late\n"
        "              layer, attention driven effect."
    )


def run_all(
    dataset,
    model_kind="ape",
    number_images=512,
    batch_size=128,
    early=(0, 1, 2, 3),
    mid=(4, 5, 6, 7),
    late=(8, 9, 10, 11),
    do_rank=False,
    do_plot=False,
    save=True,
):
    """Run every ablation condition on a pre streamed dataset and return the
    results. This is the entry point the notebook calls."""
    model, processor, source = load_model(model_kind)
    n_layers = num_blocks(model, source)
    conditions = build_conditions(n_layers, list(early), list(mid), list(late))

    rpi_curves = {}
    for name, spec in conditions:
        print(f"-- condition: {name} (RPI) --")
        rpi_curves[name] = run_condition(
            model, processor, dataset, source, spec, True, number_images, batch_size
        )
        print("   ", [round(x, 3) for x in rpi_curves[name]])

    clean_baseline = run_condition(
        model, processor, dataset, source, None, False, number_images, batch_size
    )

    rank_curves = {}
    if do_rank:
        cond_map = dict(conditions)
        for name in ["baseline", "mlp_zero_all", "attn_zero_all"]:
            print(f"-- effective rank: {name} --")
            rank_curves[name] = run_rank_condition(
                model, processor, dataset, source, cond_map[name], True, number_images, batch_size
            )

    out = {
        "model": model_kind,
        "windows": {"early": list(early), "mid": list(mid), "late": list(late)},
        "rpi": rpi_curves,
        "clean_baseline": clean_baseline,
        "rank_rpi": rank_curves,
        "summary": {name: common.summarize_curve(curve) for name, curve in rpi_curves.items()},
    }
    if save:
        common.save_json(out, os.path.join(common.RESULTS_DIR, f"ablation_layerwise_{model_kind}.json"))

    print_summary(rpi_curves)
    if do_plot:
        make_plots(rpi_curves, clean_baseline)
    return out


def main():
    parser = argparse.ArgumentParser(description="Layer windowed component ablation for the APE ViT.")
    parser.add_argument("--model", choices=["ape", "rope"], default="ape",
                        help="APE is the model the questions are about; rope is available for comparison.")
    parser.add_argument("--number-images", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--early", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--mid", type=int, nargs="+", default=[4, 5, 6, 7])
    parser.add_argument("--late", type=int, nargs="+", default=[8, 9, 10, 11])
    parser.add_argument("--rank", action="store_true", help="also record effective rank per condition")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    dataset = common.load_imagenet(token=args.hf_token)
    return run_all(
        dataset, args.model, args.number_images, args.batch_size,
        args.early, args.mid, args.late, do_rank=args.rank, do_plot=args.plot,
    )


if __name__ == "__main__":
    main()
