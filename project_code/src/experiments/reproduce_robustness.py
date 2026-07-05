"""Reproduce the robustness / fragility numbers under Gaussian blur.

Fragility = 1 - shifted_accuracy / baseline_accuracy, where the shift is
ImageNet-C Gaussian blur at severity 5. Both models classify ImageNet-1k
directly, so accuracy is read straight from the head with no finetuning.

Reference (original run): APE 0.802 / 0.541 / 0.326, RoPE 0.836 / 0.586 / 0.299
(baseline / shifted / fragility). RoPE is the more robust of the two.

Run:
    python reproduce_robustness.py --model both --number-images 1000
    python reproduce_robustness.py --model ape --corruption "JPEG"
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
from main.model import predict
from main.prep_data import prep_data


def run_model(model_kind, dataset, corruption="Gaussian Blur", severity=5, number_images=1000, batch_size=256):
    model, processor, source = load_model(model_kind)

    dl_clean = prep_data(dataset, processor, source, number_images=number_images, batch_size=batch_size)
    baseline = predict(model, dl_clean, source)

    dl_shift = prep_data(
        dataset, processor, source,
        corruption_type=corruption, severity=severity,
        number_images=number_images, batch_size=batch_size,
    )
    shifted = predict(model, dl_shift, source)

    fragility = 1.0 - shifted / baseline
    return {
        "baseline_accuracy": float(baseline),
        "shifted_accuracy": float(shifted),
        "fragility": float(fragility),
        "corruption": corruption,
        "severity": severity,
        "source": source,
    }


def main():
    parser = argparse.ArgumentParser(description="Reproduce fragility under a distribution shift.")
    parser.add_argument("--model", choices=["ape", "rope", "both"], default="both")
    parser.add_argument("--corruption", default="Gaussian Blur")
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--number-images", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    dataset = common.load_imagenet(token=args.hf_token)
    kinds = ["ape", "rope"] if args.model == "both" else [args.model]

    results = {}
    for kind in kinds:
        print(f"== {kind.upper()} : {args.corruption} (severity {args.severity}) ==")
        res = run_model(kind, dataset, args.corruption, args.severity, args.number_images, args.batch_size)
        results[kind] = res
        print(
            f"baseline {res['baseline_accuracy']:.3f}  "
            f"shifted {res['shifted_accuracy']:.3f}  "
            f"fragility {res['fragility']:.3f}"
        )

    common.save_json(results, os.path.join(common.RESULTS_DIR, "robustness_run.json"))
    return results


if __name__ == "__main__":
    main()
