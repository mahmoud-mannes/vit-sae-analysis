"""Independent layer-by-layer component ablation for APE and RoPE ViTs.

Each condition removes exactly one attention or MLP update by replacing that
component's output with zero. The residual path remains intact. We evaluate the
same intervention under normal patch order and a fixed Random Permutation at
Inference (RPI), recording final-layer SSDC and top-1 ImageNet accuracy.

This is a necessity probe: a large score drop means the ablated component was
causally required for the measured behavior in that forward pass. It complements
activation patching, which asks whether a clean component is sufficient to rescue
an already corrupted run.
"""

import argparse
import os
import sys
from contextlib import ExitStack

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import common
from interventions.ablation import AblationController
from main.load_models import get_vit_blocks, load_model, num_prefix_tokens
from metrics.ssdc import SSDCAccumulator


def build_specs(n_layers, components=("attn", "mlp"), layers=None):
    """Build one independent zero-ablation condition per component and layer."""
    layer_ids = common.validate_layer_indices(
        layers, n_layers, name="layers", default=range(n_layers)
    )
    return [
        {"name": f"{component}_L{layer:02d}", "component": component, "layer": layer}
        for component in components
        for layer in layer_ids
    ]


def _resolve_ssdc_readout_layers(n_layers, layers=None):
    """Validate requested block-output SSDC readout layers."""
    return common.validate_layer_indices(
        layers,
        n_layers,
        name="ssdc readout layers",
        default=[n_layers - 1],
    )


def run_final_readouts(model, source, batches, perm=None, half=True, ssdc_readout_layers=None):
    """Measure accuracy plus SSDC at selected block outputs.

    SSDC is captured from the post-block residual stream, matching the block
    output hooks used by activation patching. That means ablating block ``i``
    affects readout ``i`` and all later readouts.
    """
    blocks = get_vit_blocks(model, source)
    final_layer = len(blocks) - 1
    requested_layers = _resolve_ssdc_readout_layers(len(blocks), ssdc_readout_layers)
    measured_layers = list(requested_layers)
    if final_layer not in measured_layers:
        measured_layers.append(final_layer)

    accumulator = SSDCAccumulator(n_prefix=num_prefix_tokens(model, source))
    correct = total = 0

    def make_readout_hook(layer):
        def hook(module, inputs, output):
            accumulator.add(layer, common.as_tensor(output).detach())

        return hook

    with torch.inference_mode(), ExitStack() as stack:
        for layer in measured_layers:
            stack.callback(blocks[layer].register_forward_hook(make_readout_hook(layer)).remove)
        if perm is not None:
            stack.callback(common.install_fixed_rpi_hook(model, source, perm).remove)
        for pixel_values, labels in batches:
            pixel_values = common.prepare_pixel_values(pixel_values, model, half=half)
            logits = common.forward_logits(model, source, pixel_values)
            batch_correct, batch_total = common.accuracy_counts(logits, labels)
            correct += batch_correct
            total += batch_total

    final_ssdc = accumulator.ssdc(final_layer)
    ssdc_by_layer = {
        layer: accumulator.ssdc(layer) for layer in requested_layers
    }
    return {
        "final": final_ssdc,
        "ssdc_by_layer": ssdc_by_layer,
        "accuracy": correct / total,
        "n_images": total,
    }


def run_ablation_condition(model, source, batches, perm, spec, half=True, ssdc_readout_layers=None):
    """Zero one component and evaluate normal-order and RPI forward passes."""
    with AblationController(
        model,
        source,
        spec["component"],
        [spec["layer"]],
        mode="zero",
    ):
        clean = run_final_readouts(
            model,
            source,
            batches,
            perm=None,
            half=half,
            ssdc_readout_layers=ssdc_readout_layers,
        )
        rpi = run_final_readouts(
            model,
            source,
            batches,
            perm=perm,
            half=half,
            ssdc_readout_layers=ssdc_readout_layers,
        )
    return {"clean": clean, "rpi": rpi, "spec": dict(spec)}


def run_experiment(
    model_kind="ape",
    number_images=512,
    batch_size=32,
    components=("attn", "mlp"),
    layers=None,
    hf_token=None,
    dataset_id="ILSVRC/imagenet-1k",
    fallback_id="benjamin-paine/imagenet-1k-256x256",
    shuffle=False,
    sample_seed=0,
    buffer_size=2000,
    permutation_seed=0,
    half=True,
    output_suffix=None,
    ssdc_readout_layers=None,
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

    n_patches = common.patch_token_count(model, source, batches[0][0], half=half)
    perm = common.seeded_patch_permutation(n_patches, permutation_seed)
    n_layers = len(get_vit_blocks(model, source))
    resolved_ssdc_readout_layers = _resolve_ssdc_readout_layers(
        n_layers, ssdc_readout_layers
    )

    results = {
        "experiment": "independent_zero_ablation",
        "model": model_kind,
        "source": source,
        "number_images": int(sum(labels.numel() for _, labels in batches)),
        "batch_size": batch_size,
        "components": list(components),
        "layers": list(range(n_layers)) if layers is None else [int(x) for x in layers],
        "shuffle": shuffle,
        "sample_seed": sample_seed,
        "buffer_size": buffer_size,
        "permutation_seed": permutation_seed,
        "ssdc_readout_layers": resolved_ssdc_readout_layers,
        "replacement": "zero",
        "baselines": {},
        "ablations": {},
    }

    print("== baseline: normal order ==")
    results["baselines"]["clean"] = run_final_readouts(
        model,
        source,
        batches,
        perm=None,
        half=half,
        ssdc_readout_layers=resolved_ssdc_readout_layers,
    )
    print(results["baselines"]["clean"])

    print("== baseline: RPI ==")
    results["baselines"]["rpi"] = run_final_readouts(
        model,
        source,
        batches,
        perm=perm,
        half=half,
        ssdc_readout_layers=resolved_ssdc_readout_layers,
    )
    print(results["baselines"]["rpi"])

    for spec in build_specs(n_layers, components=components, layers=layers):
        print(f"== zero ablation: {spec['name']} ==")
        result = run_ablation_condition(
            model,
            source,
            batches,
            perm,
            spec,
            half=half,
            ssdc_readout_layers=resolved_ssdc_readout_layers,
        )
        results["ablations"][spec["name"]] = result
        clean, rpi = result["clean"], result["rpi"]
        print(
            f"clean acc={clean['accuracy']:.3f} final={clean['final']:.3f}; "
            f"RPI acc={rpi['accuracy']:.3f} final={rpi['final']:.3f}"
        )

    suffix = f"_{output_suffix}" if output_suffix else ""
    out_path = os.path.join(common.RESULTS_DIR, f"ablation_sweep_{model_kind}{suffix}.json")
    common.save_json(results, out_path)
    print(f"saved {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Independent layer-by-layer attention and MLP zero ablation."
    )
    parser.add_argument("--model", choices=["ape", "rope"], default="ape")
    parser.add_argument("--number-images", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--components", nargs="+", choices=["attn", "mlp"], default=["attn", "mlp"]
    )
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--dataset-id", default="ILSVRC/imagenet-1k")
    parser.add_argument("--fallback-id", default="benjamin-paine/imagenet-1k-256x256")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=2000)
    parser.add_argument("--permutation-seed", type=int, default=0)
    parser.add_argument("--ssdc-readout-layers", type=int, nargs="+", default=None)
    parser.add_argument("--no-half", action="store_true")
    parser.add_argument("--output-suffix", default=None)
    args = parser.parse_args()

    run_experiment(
        model_kind=args.model,
        number_images=args.number_images,
        batch_size=args.batch_size,
        components=tuple(args.components),
        layers=args.layers,
        hf_token=args.hf_token,
        dataset_id=args.dataset_id,
        fallback_id=args.fallback_id,
        shuffle=args.shuffle,
        sample_seed=args.sample_seed,
        buffer_size=args.buffer_size,
        permutation_seed=args.permutation_seed,
        ssdc_readout_layers=args.ssdc_readout_layers,
        half=not args.no_half,
        output_suffix=args.output_suffix,
    )
if __name__ == "__main__":
    main()
