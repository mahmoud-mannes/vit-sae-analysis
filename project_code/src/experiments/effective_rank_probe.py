"""Extension: effective rank across depth, and its collapse under MLP ablation.

Dong et al. (2021) prove that attention without MLPs and skip connections drives
token representations toward rank one with depth. This probe measures effective
rank across the residual stream for three conditions and lines the result up with
the SSDC story:

  - baseline        : intact model.
  - mlp_zero_all    : all MLP sublayers ablated (attention plus skip only).
  - attn_zero_all   : all attention sublayers ablated (MLP plus skip only).

The prediction is that mlp_zero_all lets effective rank fall with depth (the
Dong et al. collapse), and that this fall coincides with the SSDC recovery loss
measured in ablation_layerwise.py. That links a representational capacity failure
to the spatial structure failure, giving a mechanistic account rather than a
correlation.

Run:
    python effective_rank_probe.py --model ape --plot
    python effective_rank_probe.py --model rope --plot --center
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
from metrics.effective_rank import evaluate_effective_rank


def run(model_kind, dataset, number_images=512, batch_size=128, center=False, rpi=False):
    model, processor, source = load_model(model_kind)
    n_layers = num_blocks(model, source)
    all_layers = list(range(n_layers))
    conditions = {
        "baseline": None,
        "mlp_zero_all": ("mlp", all_layers, "zero"),
        "attn_zero_all": ("attn", all_layers, "zero"),
    }
    curves = {}
    for name, spec in conditions.items():
        print(f"-- effective rank: {name} --")
        if spec is None:
            curves[name] = evaluate_effective_rank(
                model, processor, dataset, source, RPI=rpi,
                number_images=number_images, batch_size=batch_size, center=center,
            )
        else:
            component, layers, mode = spec
            with AblationController(model, source, component, layers, mode):
                curves[name] = evaluate_effective_rank(
                    model, processor, dataset, source, RPI=rpi,
                    number_images=number_images, batch_size=batch_size, center=center,
                )
        print("   ", [round(x, 2) for x in curves[name]])
    return curves


def main():
    parser = argparse.ArgumentParser(description="Effective rank across depth under ablation.")
    parser.add_argument("--model", choices=["ape", "rope"], default="ape")
    parser.add_argument("--number-images", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--center", action="store_true", help="subtract the mean token first (collapse-toward-common view)")
    parser.add_argument("--rpi", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    dataset = common.load_imagenet(token=args.hf_token)
    curves = run(args.model, dataset, args.number_images, args.batch_size, args.center, args.rpi)

    common.save_json(
        {"model": args.model, "center": args.center, "rpi": args.rpi, "effective_rank": curves},
        os.path.join(common.RESULTS_DIR, f"effective_rank_{args.model}.json"),
    )
    if args.plot:
        common.plot_curves(
            curves,
            title=f"Effective rank across depth: {args.model.upper()}",
            ylabel="effective rank",
            save_path=os.path.join(common.FIGURES_DIR, f"effective_rank_{args.model}.png"),
        )
    return curves


if __name__ == "__main__":
    main()
