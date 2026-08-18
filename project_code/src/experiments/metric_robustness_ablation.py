"""Test whether causal SSDC localization depends on cosine similarity.

The established SSDC analyses compare cosine similarity between patch-token
representations with spatial distance. This experiment repeats selected causal
ablations with two controls: cosine after removing each image's token mean, and
negative Euclidean distance. Agreement across the metrics makes a metric-specific
anisotropy or common-direction explanation less likely.
"""

import argparse
import os
import re
import sys
from contextlib import ExitStack

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import common
from interventions.ablation import AblationController
from main.load_models import get_vit_blocks, load_model, num_prefix_tokens
from metrics.ssdc import PAIRWISE_SIMILARITY_METRICS, SSDCAccumulator


METRIC_NAMES = PAIRWISE_SIMILARITY_METRICS


def parse_specs(spec_names, n_layers):
    """Parse names such as ``attn_L03`` into ablation specifications."""
    specs = []
    for name in spec_names:
        match = re.fullmatch(r"(attn|mlp)_L(\d+)", name)
        if match is None:
            raise ValueError(f"invalid ablation spec {name!r}; expected attn_L03 or mlp_L03")
        component, layer_text = match.groups()
        layer = int(layer_text)
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"layer {layer} is out of range for a {n_layers}-layer model")
        specs.append({"name": f"{component}_L{layer:02d}", "component": component, "layer": layer})
    return specs


def _resolve_readout_layers(n_layers, layers):
    return common.validate_layer_indices(layers, n_layers, name="readout layers")


def run_multi_metric_readouts(model, source, batches, readout_layers, perm=None, half=True):
    """Measure accuracy and all three SSDC variants at selected block outputs."""
    blocks = get_vit_blocks(model, source)
    layers = _resolve_readout_layers(len(blocks), readout_layers)
    n_prefix = num_prefix_tokens(model, source)
    accumulator = SSDCAccumulator(
        metrics=METRIC_NAMES,
        n_prefix=0,
    )
    correct = total = 0

    def make_hook(layer):
        def hook(module, inputs, output):
            patch_tokens = common.as_tensor(output).detach()[:, n_prefix:]
            accumulator.add(layer, patch_tokens)

        return hook

    with torch.inference_mode(), ExitStack() as stack:
        for layer in layers:
            stack.callback(blocks[layer].register_forward_hook(make_hook(layer)).remove)
        if perm is not None:
            stack.callback(common.install_fixed_rpi_hook(model, source, perm).remove)
        for pixel_values, labels in batches:
            pixel_values = common.prepare_pixel_values(pixel_values, model, half=half)
            logits = common.forward_logits(model, source, pixel_values)
            batch_correct, batch_total = common.accuracy_counts(logits, labels)
            correct += batch_correct
            total += batch_total

    return {
        "metrics_by_layer": {
            layer: {
                metric_name: accumulator.ssdc(layer, metric=metric_name)
                for metric_name in METRIC_NAMES
            }
            for layer in layers
        },
        "accuracy": correct / total,
        "n_images": total,
    }


def run_ablation_condition(model, source, batches, perm, spec, readout_layers, half=True):
    with AblationController(model, source, spec["component"], [spec["layer"]], mode="zero"):
        clean = run_multi_metric_readouts(
            model, source, batches, readout_layers, perm=None, half=half
        )
        rpi = run_multi_metric_readouts(
            model, source, batches, readout_layers, perm=perm, half=half
        )
    return {"clean": clean, "rpi": rpi, "spec": dict(spec)}


def run_experiment(
    model_kind,
    spec_names,
    readout_layers=(4, 5),
    number_images=512,
    batch_size=32,
    hf_token=None,
    dataset_id="ILSVRC/imagenet-1k",
    fallback_id="benjamin-paine/imagenet-1k-256x256",
    shuffle=False,
    sample_seed=0,
    buffer_size=2000,
    permutation_seed=0,
    half=True,
    output_suffix=None,
):
    common.ensure_dirs()
    model, processor, source = load_model(model_kind, half=half)
    dataset = common.load_imagenet(
        token=hf_token,
        dataset_id=dataset_id,
        fallback_id=fallback_id,
        streaming=True,
        shuffle=shuffle,
        seed=sample_seed,
        buffer_size=buffer_size,
    )
    batches = common.collect_batches(dataset, processor, source, number_images, batch_size)
    if not batches:
        raise RuntimeError("no batches collected from dataset")

    blocks = get_vit_blocks(model, source)
    resolved_layers = _resolve_readout_layers(len(blocks), readout_layers)
    specs = parse_specs(spec_names, len(blocks))
    n_patches = common.patch_token_count(model, source, batches[0][0], half=half)
    perm = common.seeded_patch_permutation(n_patches, permutation_seed)

    results = {
        "experiment": "metric_robustness_zero_ablation",
        "model": model_kind,
        "source": source,
        "number_images": int(sum(labels.numel() for _, labels in batches)),
        "batch_size": batch_size,
        "metrics": list(METRIC_NAMES),
        "readout_layers": resolved_layers,
        "specs": specs,
        "shuffle": shuffle,
        "sample_seed": sample_seed,
        "buffer_size": buffer_size,
        "permutation_seed": permutation_seed,
        "replacement": "zero",
        "baselines": {},
        "ablations": {},
    }

    for order_name, order_perm in (("clean", None), ("rpi", perm)):
        print(f"== baseline: {order_name} ==")
        results["baselines"][order_name] = run_multi_metric_readouts(
            model, source, batches, resolved_layers, perm=order_perm, half=half
        )
        print(results["baselines"][order_name])

    for spec in specs:
        print(f"== zero ablation: {spec['name']} ==")
        result = run_ablation_condition(
            model, source, batches, perm, spec, resolved_layers, half=half
        )
        results["ablations"][spec["name"]] = result
        print(result)

    suffix = f"_{output_suffix}" if output_suffix else ""
    out_path = os.path.join(
        common.RESULTS_DIR, f"metric_robustness_ablation_{model_kind}{suffix}.json"
    )
    common.save_json(results, out_path)
    print(f"saved {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["ape", "rope"], required=True)
    parser.add_argument("--specs", nargs="+", required=True)
    parser.add_argument("--readout-layers", type=int, nargs="+", default=[4, 5])
    parser.add_argument("--number-images", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--dataset-id", default="ILSVRC/imagenet-1k")
    parser.add_argument("--fallback-id", default="benjamin-paine/imagenet-1k-256x256")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=2000)
    parser.add_argument("--permutation-seed", type=int, default=0)
    parser.add_argument("--no-half", action="store_true")
    parser.add_argument("--output-suffix", default=None)
    args = parser.parse_args()

    run_experiment(
        model_kind=args.model,
        spec_names=args.specs,
        readout_layers=args.readout_layers,
        number_images=args.number_images,
        batch_size=args.batch_size,
        hf_token=args.hf_token,
        dataset_id=args.dataset_id,
        fallback_id=args.fallback_id,
        shuffle=args.shuffle,
        sample_seed=args.sample_seed,
        buffer_size=args.buffer_size,
        permutation_seed=args.permutation_seed,
        half=not args.no_half,
        output_suffix=args.output_suffix,
    )
if __name__ == "__main__":
    main()
