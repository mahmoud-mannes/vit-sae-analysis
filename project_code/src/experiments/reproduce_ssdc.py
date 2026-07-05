"""Reproduce the per layer SSDC curves, clean and under RPI, for both models.

APE (google/vit-base-patch16-224) and RoPE (vit_base_patch16_rope_224.naver_in1k)
are ViT-Base/16 models trained on ImageNet-1k. We stream the ImageNet-1k
validation split, capture the residual stream entering each of the 12 blocks, and
compute SSDC with and without Random Permutation at Inference.

Expected shape of the result (this is what the reference curves show):
  - APE clean SSDC rises through the early and middle blocks and stays high.
  - APE SSDC under RPI peaks early (around blocks 2 to 3) then decays with depth.
  - RoPE SSDC under RPI starts near zero and accumulates more gradually, peaking
    later and mid stack.

Run:
    python reproduce_ssdc.py --model ape --number-images 1000
    python reproduce_ssdc.py --model rope --number-images 1000
    python reproduce_ssdc.py --model both --plot
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
from main.load_models import load_model
from metrics.ssdc import evaluate_ssdc


def run_model(model_kind, dataset, number_images=1000, batch_size=256):
    model, processor, source = load_model(model_kind)
    clean, _ = evaluate_ssdc(
        model, processor, dataset, source, RPI=False,
        number_images=number_images, batch_size=batch_size,
    )
    rpi, _ = evaluate_ssdc(
        model, processor, dataset, source, RPI=True,
        number_images=number_images, batch_size=batch_size,
    )
    return {"clean": clean, "rpi": rpi, "source": source}


def _reference_for(model_kind):
    ref_path = os.path.join(common.REFERENCE_DIR, "ssdc_reference.json")
    if not os.path.exists(ref_path):
        return None
    ref = common.load_json(ref_path)["models"]
    key = "ape_vit_base_patch16_google" if model_kind == "ape" else "rope_vit_base_patch16_timm"
    return ref.get(key)


def plot_model(model_kind, result, save=True):
    ref = _reference_for(model_kind)
    curves = {"clean (this run)": result["clean"], "RPI (this run)": result["rpi"]}
    styles = {
        "clean (this run)": {"color": "tab:blue"},
        "RPI (this run)": {"color": "tab:orange"},
    }
    if ref is not None:
        curves["clean (reference)"] = ref["clean"]
        curves["RPI (reference)"] = ref["rpi"]
        styles["clean (reference)"] = {"color": "tab:blue", "linestyle": "--", "alpha": 0.6}
        styles["RPI (reference)"] = {"color": "tab:orange", "linestyle": "--", "alpha": 0.6}
    save_path = os.path.join(common.FIGURES_DIR, f"ssdc_{model_kind}.png") if save else None
    common.plot_curves(
        curves, title=f"SSDC across depth: {model_kind.upper()} ViT-Base/16",
        ylabel="SSDC", save_path=save_path, styles=styles,
    )


def main():
    parser = argparse.ArgumentParser(description="Reproduce SSDC and SSDC under RPI.")
    parser.add_argument("--model", choices=["ape", "rope", "both"], default="both")
    parser.add_argument("--number-images", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    dataset = common.load_imagenet(token=args.hf_token)
    kinds = ["ape", "rope"] if args.model == "both" else [args.model]

    all_results = {}
    for kind in kinds:
        print(f"== {kind.upper()} ==")
        result = run_model(kind, dataset, args.number_images, args.batch_size)
        all_results[kind] = result
        print("clean:", [round(x, 4) for x in result["clean"]])
        print("rpi:  ", [round(x, 4) for x in result["rpi"]])
        if args.plot:
            plot_model(kind, result)

    common.save_json(all_results, os.path.join(common.RESULTS_DIR, "ssdc_run.json"))
    return all_results


if __name__ == "__main__":
    main()
