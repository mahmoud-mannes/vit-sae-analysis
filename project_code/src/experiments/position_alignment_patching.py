"""Causal control experiments for positional alignment under RPI.

This experiment builds on activation patching and compares four conditions for
each selected layer/component spec:

  1. RPI baseline
  2. same-image clean patch (aligned)
  3. same-image clean patch with patch-token positions permuted while the class
     token stays fixed (misaligned)
  4. clean activation deranged across images within the batch
  5. clean activation replaced by the batch mean donor

The primary positional contrast is aligned versus token-misaligned donors. A
different-image donor preserves token alignment while changing image content;
matching the aligned donor on SSDC would therefore support image-independent
spatial information. Accuracy is recorded separately to reveal content effects.
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
from experiments.activation_patching import (
    block_output_ssdc_scores,
    build_specs,
    install_block_output_ssdc_hooks,
    install_clean_cache_hook,
    install_patch_hook,
    run_baseline,
)
from main.load_models import get_vit_blocks, load_model, num_prefix_tokens
from metrics.ssdc import SSDCAccumulator


CONTROL_ALIGNED = "same_image_aligned"
CONTROL_MISALIGNED = "same_image_misaligned"
CONTROL_DIFFERENT_IMAGE = "different_image_deranged"
CONTROL_MEAN_DONOR = "mean_donor"

DEFAULT_STRONG_SPECS = {
    "ape": ("mlp_L00", "attn_L00", "attn_L02", "attn_L03"),
    # Inferred from the existing experiment summary: RoPE effects are distributed
    # around the mid-depth readout regime rather than sharply localized at layer 0.
    "rope": ("attn_L05", "mlp_L05", "attn_L07"),
}


def _validate_readout_layers(readout_layers, n_layers):
    return common.validate_layer_indices(
        readout_layers, n_layers, name="readout layers"
    )


def _build_patch_token_permutation(n_patches, seed):
    return common.seeded_patch_permutation(n_patches, seed)


def _build_batch_derangement(batch_size, seed):
    batch_size = int(batch_size)
    if batch_size <= 1:
        raise ValueError("different-image control requires batch size > 1")
    shift = (int(seed) % (batch_size - 1)) + 1
    return torch.remainder(torch.arange(batch_size), batch_size).roll(-shift)


def _mean_donor(value):
    donor = value.mean(dim=0, keepdim=True)
    return donor.expand_as(value).clone()


def _token_axis(spec, value):
    if spec.get("component") == "qkv" and value.ndim == 4 and value.shape[1] == spec.get("n_heads"):
        return 2
    return 1


def _apply_fixed_cls_permutation(value, spec, patch_permutation):
    token_axis = _token_axis(spec, value)
    token_count = int(value.shape[token_axis])
    if token_count <= 1:
        return value.clone()
    if patch_permutation.numel() != token_count - 1:
        raise ValueError(
            f"patch permutation has length {patch_permutation.numel()} but expected {token_count - 1}"
        )
    prefix = torch.zeros(1, dtype=torch.long)
    full_index = torch.cat((prefix, patch_permutation.to(dtype=torch.long) + 1))
    return value.index_select(token_axis, full_index.to(value.device))


def _apply_batch_derangement(value, derangement):
    if derangement.numel() != value.shape[0]:
        raise ValueError(
            f"batch derangement has length {derangement.numel()} but expected batch size {value.shape[0]}"
        )
    return value.index_select(0, derangement.to(device=value.device, dtype=torch.long))


def _transform_cached_activation(value, spec, control, patch_permutation=None, batch_derangement=None):
    if control == CONTROL_ALIGNED:
        return value
    if control == CONTROL_MISALIGNED:
        if patch_permutation is None:
            raise ValueError("misaligned control requires a patch permutation")
        return _apply_fixed_cls_permutation(value, spec, patch_permutation)
    if control == CONTROL_DIFFERENT_IMAGE:
        if batch_derangement is None:
            raise ValueError("different-image control requires a batch derangement")
        return _apply_batch_derangement(value, batch_derangement)
    if control == CONTROL_MEAN_DONOR:
        return _mean_donor(value)
    raise ValueError(f"unknown control {control!r}")


def _readout_scores(scores, readout_layers):
    return {str(layer): float(scores[layer]) for layer in readout_layers}


def _summarize_condition(scores, correct, total, readout_layers):
    out = common.summarize_ssdc(scores)
    out["accuracy"] = correct / total
    out["n_images"] = total
    out["readout_layers"] = _readout_scores(scores, readout_layers)
    return out


def _augment_baseline(result, readout_layers):
    out = dict(result)
    out["scores"] = [float(score) for score in result["scores"]]
    out["readout_layers"] = _readout_scores(out["scores"], readout_layers)
    return out


def _resolve_specs(model, source, model_kind, spec_names=None):
    all_specs = build_specs(
        model,
        source,
        ("residual", "attn", "mlp"),
        include_heads=True,
        include_qkv=True,
    )
    by_name = {spec["name"]: spec for spec in all_specs}
    requested = list(DEFAULT_STRONG_SPECS[model_kind] if spec_names is None else spec_names)
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(f"unknown specs requested: {missing}")
    return [by_name[name] for name in requested]


def _control_contrasts(spec_results, readout_layers):
    aligned = spec_results[CONTROL_ALIGNED]["readout_layers"]
    misaligned = spec_results[CONTROL_MISALIGNED]["readout_layers"]
    different_image = spec_results[CONTROL_DIFFERENT_IMAGE]["readout_layers"]
    mean_donor = spec_results[CONTROL_MEAN_DONOR]["readout_layers"]
    out = {}
    for layer in readout_layers:
        key = str(layer)
        out[key] = {
            "aligned_minus_misaligned": float(aligned[key] - misaligned[key]),
            "aligned_minus_different_image": float(aligned[key] - different_image[key]),
            "different_image_minus_misaligned": float(
                different_image[key] - misaligned[key]
            ),
            "mean_donor_minus_misaligned": float(mean_donor[key] - misaligned[key]),
            "aligned_gt_misaligned": bool(aligned[key] > misaligned[key]),
            "different_image_gt_misaligned": bool(
                different_image[key] > misaligned[key]
            ),
        }
    out["accuracy"] = {
        "aligned_minus_different_image": float(
            spec_results[CONTROL_ALIGNED]["accuracy"]
            - spec_results[CONTROL_DIFFERENT_IMAGE]["accuracy"]
        ),
        "aligned_minus_misaligned": float(
            spec_results[CONTROL_ALIGNED]["accuracy"]
            - spec_results[CONTROL_MISALIGNED]["accuracy"]
        ),
    }
    return out


def run_control_condition(
    model,
    source,
    batches,
    perm,
    spec,
    control,
    readout_layers,
    patch_permutation=None,
    derangement_seed=0,
    half=True,
):
    accumulator = SSDCAccumulator(n_prefix=num_prefix_tokens(model, source))
    correct = total = 0
    with torch.inference_mode():
        for pixel_values, labels in batches:
            pixel_values = common.prepare_pixel_values(pixel_values, model, half=half)

            cache = {}
            clean_handle = install_clean_cache_hook(model, source, spec, cache)
            try:
                common.forward_logits(model, source, pixel_values)
            finally:
                clean_handle.remove()

            batch_derangement = None
            if control == CONTROL_DIFFERENT_IMAGE:
                batch_derangement = _build_batch_derangement(labels.numel(), derangement_seed)
            cache["value"] = _transform_cached_activation(
                cache["value"],
                spec,
                control,
                patch_permutation=patch_permutation,
                batch_derangement=batch_derangement,
            )

            with ExitStack() as stack:
                stack.callback(common.install_fixed_rpi_hook(model, source, perm).remove)
                stack.callback(install_patch_hook(model, source, spec, cache).remove)
                for handle in install_block_output_ssdc_hooks(
                    model, source, accumulator
                ):
                    stack.callback(handle.remove)
                logits = common.forward_logits(model, source, pixel_values)

            c, t = common.accuracy_counts(logits, labels)
            correct += c
            total += t

    scores = block_output_ssdc_scores(accumulator)
    return _summarize_condition(scores, correct, total, readout_layers)


def _filter_control_batches(batches):
    usable = [(images, labels) for images, labels in batches if labels.numel() > 1]
    dropped = len(batches) - len(usable)
    if not usable:
        raise ValueError("different-image control requires at least one batch with more than one image")
    return usable, dropped


def run_experiment(
    model_kind="ape",
    number_images=128,
    batch_size=32,
    spec_names=None,
    readout_layers=(4, 5),
    hf_token=None,
    dataset_id="ILSVRC/imagenet-1k",
    fallback_id="benjamin-paine/imagenet-1k-256x256",
    corruption_type=None,
    severity=5,
    shuffle=False,
    sample_seed=0,
    buffer_size=2000,
    permutation_seed=0,
    control_permutation_seed=1,
    derangement_seed=0,
    half=True,
    output_suffix=None,
):
    if int(batch_size) <= 1:
        raise ValueError("different-image control requires batch size > 1")

    common.ensure_dirs()
    model, processor, source = load_model(model_kind, half=half)
    n_layers = len(get_vit_blocks(model, source))
    readout_layers = _validate_readout_layers(readout_layers, n_layers)

    dataset = common.load_imagenet(
        token=hf_token,
        dataset_id=dataset_id,
        fallback_id=fallback_id,
        streaming=True,
        shuffle=shuffle,
        seed=sample_seed,
        buffer_size=buffer_size,
    )
    batches = common.collect_batches(
        dataset,
        processor,
        source,
        number_images,
        batch_size,
        corruption_type=corruption_type,
        severity=severity,
    )
    if not batches:
        raise RuntimeError("no batches collected from dataset")
    control_batches, dropped_singleton_batches = _filter_control_batches(batches)

    first = control_batches[0][0]
    n_patches = common.patch_token_count(model, source, first, half=half)
    rpi_perm = _build_patch_token_permutation(n_patches, permutation_seed)
    control_perm = _build_patch_token_permutation(n_patches, control_permutation_seed)

    results = {
        "model": model_kind,
        "source": source,
        "number_images": int(sum(labels.numel() for _, labels in control_batches)),
        "batch_size": int(batch_size),
        "spec_names": list(DEFAULT_STRONG_SPECS[model_kind] if spec_names is None else spec_names),
        "readout_layers": list(readout_layers),
        "corruption_type": corruption_type,
        "severity": int(severity),
        "shuffle": bool(shuffle),
        "sample_seed": int(sample_seed),
        "buffer_size": int(buffer_size),
        "permutation_seed": int(permutation_seed),
        "control_permutation_seed": int(control_permutation_seed),
        "derangement_seed": int(derangement_seed),
        "dropped_singleton_batches": int(dropped_singleton_batches),
        "baselines": {},
        "controls": {},
    }

    print("== baseline: clean ==")
    clean = run_baseline(model, source, control_batches, perm=None, half=half)
    results["baselines"]["clean"] = _augment_baseline(clean, readout_layers)
    print(results["baselines"]["clean"])

    print("== baseline: rpi ==")
    rpi = run_baseline(model, source, control_batches, perm=rpi_perm, half=half)
    results["baselines"]["rpi"] = _augment_baseline(rpi, readout_layers)
    print(results["baselines"]["rpi"])

    for spec in _resolve_specs(model, source, model_kind, spec_names=spec_names):
        print(f"== controls: {spec['name']} ==")
        spec_results = {"spec": {k: v for k, v in spec.items() if k != "n_heads"}}
        for control in (
            CONTROL_ALIGNED,
            CONTROL_MISALIGNED,
            CONTROL_DIFFERENT_IMAGE,
            CONTROL_MEAN_DONOR,
        ):
            result = run_control_condition(
                model,
                source,
                control_batches,
                rpi_perm,
                spec,
                control,
                readout_layers,
                patch_permutation=control_perm if control == CONTROL_MISALIGNED else None,
                derangement_seed=derangement_seed,
                half=half,
            )
            spec_results[control] = result
            print(
                f"{control}: acc={result['accuracy']:.3f} "
                f"readouts={result['readout_layers']}"
            )
        spec_results["control_contrasts"] = _control_contrasts(
            spec_results, readout_layers
        )
        results["controls"][spec["name"]] = spec_results

    suffix = f"_{output_suffix}" if output_suffix else ""
    out_path = os.path.join(common.RESULTS_DIR, f"position_alignment_patching_{model_kind}{suffix}.json")
    common.save_json(results, out_path)
    print(f"saved {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Causal control experiment for ViT positional alignment.")
    parser.add_argument("--model", choices=["ape", "rope"], default="ape")
    parser.add_argument("--number-images", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--specs", nargs="+", default=None)
    parser.add_argument("--readout-layers", type=int, nargs="+", default=[4, 5])
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--dataset-id", default="ILSVRC/imagenet-1k")
    parser.add_argument("--fallback-id", default="benjamin-paine/imagenet-1k-256x256")
    parser.add_argument("--corruption", default=None)
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=2000)
    parser.add_argument("--permutation-seed", type=int, default=0)
    parser.add_argument("--control-permutation-seed", type=int, default=1)
    parser.add_argument("--derangement-seed", type=int, default=0)
    parser.add_argument("--no-half", action="store_true")
    parser.add_argument("--output-suffix", default=None)
    args = parser.parse_args()

    run_experiment(
        model_kind=args.model,
        number_images=args.number_images,
        batch_size=args.batch_size,
        spec_names=args.specs,
        readout_layers=args.readout_layers,
        hf_token=args.hf_token,
        dataset_id=args.dataset_id,
        fallback_id=args.fallback_id,
        corruption_type=args.corruption,
        severity=args.severity,
        shuffle=args.shuffle,
        sample_seed=args.sample_seed,
        buffer_size=args.buffer_size,
        permutation_seed=args.permutation_seed,
        control_permutation_seed=args.control_permutation_seed,
        derangement_seed=args.derangement_seed,
        half=not args.no_half,
        output_suffix=args.output_suffix,
    )
if __name__ == "__main__":
    main()
