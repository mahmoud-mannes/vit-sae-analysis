"""Block-stream SSDC and attention-output probe analysis for ViTs.

This script measures how much index-anchored spatial structure is carried or
amplified by explicit block stages and raw sublayer outputs:

1. Block input (the residual stream entering attention).
2. Post-attention residual (after adding the attention update, before the MLP).
3. Block output (after the MLP residual update).
4. Raw attention output.
5. Raw MLP output contribution.

For each stream we report SSDC on clean inputs and under Random Permutation at
Inference (RPI). We also report the residual-stage deltas
`post_attention_residual - block_input` and
`block_output - post_attention_residual`. Separately, we fit held-out ridge
probes on the attention output under RPI to measure linear decodability of
normalized patch row/column coordinates. Those probes train and evaluate on
disjoint image splits and on disjoint RPI permutation seeds, with a
shuffled-label negative control.

Run:
    python attention_output_analysis.py --model both --number-images 1000
    python attention_output_analysis.py --model ape --batch-size 128 --half
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
from main.load_models import (
    get_block_attention,
    get_block_mlp,
    get_vit_blocks,
    load_model,
    num_prefix_tokens,
)
from metrics.position_probe import (
    DEFAULT_COORDINATE_PROBE_ALPHA_GRID,
    evaluate_coordinate_probe,
    split_image_indices,
)
from metrics.ssdc import SSDCAccumulator

STREAM_NAMES = (
    "block_input",
    "post_attention_residual",
    "block_output",
    "attention_output",
    "mlp_output",
)


def _build_patch_permutation(model, source, sample_batch, seed, half=False):
    num_patches = common.patch_token_count(
        model, source, sample_batch, half=half
    )
    return common.seeded_patch_permutation(num_patches, seed)


def _accumulate_similarity(accumulator, stream_name, layer_idx, tokens):
    accumulator.add((stream_name, layer_idx), tokens.detach())


def resolve_block_stage_topology(block, source):
    """Return the modules that expose the stages needed for SSDC capture."""

    if source == "timm":
        post_attention_norm = block.norm2
    elif hasattr(block, "layernorm_after"):
        post_attention_norm = block.layernorm_after
    else:
        raise AttributeError("could not locate the post-attention normalization module")

    mlp_module, mlp_mode = get_block_mlp(block, source)
    return {
        "attention_module": get_block_attention(block, source),
        "post_attention_norm": post_attention_norm,
        "mlp_output_module": mlp_module,
        "mlp_output_mode": mlp_mode,
    }


def _make_block_input_hook(store, layer_idx):
    def hook(module, inputs):
        _accumulate_similarity(store, "block_input", layer_idx, inputs[0])

    return hook


def _make_post_attention_residual_hook(store, layer_idx):
    def hook(module, inputs):
        _accumulate_similarity(store, "post_attention_residual", layer_idx, inputs[0])

    return hook


def _make_block_output_hook(store, layer_idx):
    def hook(module, inputs, output):
        _accumulate_similarity(store, "block_output", layer_idx, common.as_tensor(output))

    return hook


def _make_attention_output_hook(store, layer_idx):
    def hook(module, inputs, output):
        _accumulate_similarity(store, "attention_output", layer_idx, common.as_tensor(output))

    return hook


def _make_mlp_stream_hook(store, layer_idx, mode):
    def hook(module, inputs, output):
        mlp_out = common.as_tensor(output)
        if mode == "residual":
            mlp_out = mlp_out - inputs[1]
        _accumulate_similarity(store, "mlp_output", layer_idx, mlp_out)

    return hook


def capture_block_stream_ssdc(
    model,
    source,
    batches,
    perm=None,
    half=False,
    metric="manhattan",
):
    """Return SSDC curves for explicit block stages and raw sublayer outputs."""

    n_prefix = num_prefix_tokens(model, source)
    accumulator = SSDCAccumulator(n_prefix=n_prefix, spatial_metric=metric)
    handles = []
    for layer_idx, block in enumerate(get_vit_blocks(model, source)):
        topology = resolve_block_stage_topology(block, source)
        handles.append(
            block.register_forward_pre_hook(
                _make_block_input_hook(accumulator, layer_idx)
            )
        )
        handles.append(
            topology["post_attention_norm"].register_forward_pre_hook(
                _make_post_attention_residual_hook(accumulator, layer_idx)
            )
        )
        handles.append(
            block.register_forward_hook(_make_block_output_hook(accumulator, layer_idx))
        )
        handles.append(
            topology["attention_module"].register_forward_hook(
                _make_attention_output_hook(accumulator, layer_idx)
            )
        )
        handles.append(
            topology["mlp_output_module"].register_forward_hook(
                _make_mlp_stream_hook(
                    accumulator, layer_idx, topology["mlp_output_mode"]
                )
            )
        )

    with torch.inference_mode(), ExitStack() as stack:
        for handle in handles:
            stack.callback(handle.remove)
        if perm is not None:
            stack.callback(common.install_fixed_rpi_hook(model, source, perm).remove)
        for pixel_values, _labels in batches:
            pixel_values = common.prepare_pixel_values(pixel_values, model, half=half)
            common.forward_logits(model, source, pixel_values)

    def stream_scores(stream_name):
        return [
            accumulator.ssdc((stream_name, layer_idx))
            for layer_idx in range(len(get_vit_blocks(model, source)))
        ]

    block_input = stream_scores("block_input")
    post_attention_residual = stream_scores("post_attention_residual")
    block_output = stream_scores("block_output")
    attention_output = stream_scores("attention_output")
    mlp_output = stream_scores("mlp_output")
    return {
        "block_input": block_input,
        "post_attention_residual": post_attention_residual,
        "block_output": block_output,
        "attention_output": attention_output,
        "mlp_output": mlp_output,
        "attention_residual_delta": [
            float(a - b) for a, b in zip(post_attention_residual, block_input)
        ],
        "mlp_residual_delta": [
            float(a - b) for a, b in zip(block_output, post_attention_residual)
        ],
        "mlp_residual_delta_well_defined": True,
    }


def capture_attention_output_ssdc(model, source, batches, perm=None, half=False, metric="manhattan"):
    """Backward-compatible wrapper returning only the attention-output SSDC curve."""

    return capture_block_stream_ssdc(
        model,
        source,
        batches,
        perm=perm,
        half=half,
        metric=metric,
    )["attention_output"]


def collect_layer_attention_outputs(model, source, batches, layer_idx, perm=None, half=False):
    """Capture one layer's attention outputs for all images as CPU tensors."""

    outputs = []
    n_prefix = num_prefix_tokens(model, source)
    attn = get_block_attention(get_vit_blocks(model, source)[layer_idx], source)

    def hook(module, inputs, output):
        tok = common.as_tensor(output).detach()[:, n_prefix:, :].to("cpu", dtype=torch.float32).clone()
        outputs.append(tok)

    handle = attn.register_forward_hook(hook)
    with torch.inference_mode(), ExitStack() as stack:
        stack.callback(handle.remove)
        if perm is not None:
            stack.callback(common.install_fixed_rpi_hook(model, source, perm).remove)
        for pixel_values, _labels in batches:
            pixel_values = common.prepare_pixel_values(pixel_values, model, half=half)
            common.forward_logits(model, source, pixel_values)

    return torch.cat(outputs, dim=0).numpy()


def run_model_experiment(
    model_kind,
    dataset,
    number_images=1000,
    batch_size=128,
    sample_seed=0,
    permutation_seed=0,
    half=False,
    alpha_grid=DEFAULT_COORDINATE_PROBE_ALPHA_GRID,
):
    applied_half = bool(half and torch.cuda.is_available())
    model, processor, source = load_model(model_kind, half=applied_half)
    batches = common.collect_batches(dataset, processor, source, number_images, batch_size)
    if not batches:
        raise ValueError("no batches were collected from the dataset")

    total_images = int(sum(labels.numel() for _, labels in batches))
    split = split_image_indices(total_images, seed=sample_seed)
    rpi_perm = _build_patch_permutation(model, source, batches[0][0], seed=permutation_seed, half=applied_half)
    probe_eval_seed = permutation_seed + 1
    probe_eval_perm = _build_patch_permutation(
        model,
        source,
        batches[0][0],
        seed=probe_eval_seed,
        half=applied_half,
    )

    clean_streams = capture_block_stream_ssdc(model, source, batches, perm=None, half=applied_half)
    rpi_streams = capture_block_stream_ssdc(model, source, batches, perm=rpi_perm, half=applied_half)

    layer_metrics = []
    n_layers = len(get_vit_blocks(model, source))
    for layer_idx in range(n_layers):
        train_outputs = collect_layer_attention_outputs(
            model,
            source,
            batches,
            layer_idx,
            perm=rpi_perm,
            half=applied_half,
        )
        eval_outputs = collect_layer_attention_outputs(
            model,
            source,
            batches,
            layer_idx,
            perm=probe_eval_perm,
            half=applied_half,
        )
        probe = evaluate_coordinate_probe(
            train_outputs,
            eval_outputs,
            split,
            shuffle_seed=permutation_seed + 1009 * (layer_idx + 1),
            alpha_grid=alpha_grid,
        )
        layer_metrics.append(
            {
                "layer": int(layer_idx),
                "block_input_ssdc": {
                    "clean": float(clean_streams["block_input"][layer_idx]),
                    "rpi": float(rpi_streams["block_input"][layer_idx]),
                },
                "post_attention_residual_ssdc": {
                    "clean": float(clean_streams["post_attention_residual"][layer_idx]),
                    "rpi": float(rpi_streams["post_attention_residual"][layer_idx]),
                },
                "block_output_ssdc": {
                    "clean": float(clean_streams["block_output"][layer_idx]),
                    "rpi": float(rpi_streams["block_output"][layer_idx]),
                },
                "attention_output_ssdc": {
                    "clean": float(clean_streams["attention_output"][layer_idx]),
                    "rpi": float(rpi_streams["attention_output"][layer_idx]),
                },
                "mlp_output_ssdc": {
                    "clean": float(clean_streams["mlp_output"][layer_idx]),
                    "rpi": float(rpi_streams["mlp_output"][layer_idx]),
                },
                "attention_residual_delta": {
                    "clean": float(clean_streams["attention_residual_delta"][layer_idx]),
                    "rpi": float(rpi_streams["attention_residual_delta"][layer_idx]),
                },
                "mlp_residual_delta": {
                    "clean": float(clean_streams["mlp_residual_delta"][layer_idx]),
                    "rpi": float(rpi_streams["mlp_residual_delta"][layer_idx]),
                },
                "probe_rpi": probe,
            }
        )

    return {
        "model": model_kind,
        "source": source,
        "n_layers": int(n_layers),
        "n_prefix_tokens": int(num_prefix_tokens(model, source)),
        "number_images": int(total_images),
        "batch_size": int(batch_size),
        "sample_seed": int(sample_seed),
        "permutation_seed": int(permutation_seed),
        "requested_half": bool(half),
        "applied_half": bool(applied_half),
        "alpha_grid": [float(alpha) for alpha in alpha_grid],
        "claims": [
            "Residual-stage SSDC deltas indicate whether attention or MLP updates "
            "carry or amplify index-anchored spatial structure across explicit "
            "block stages.",
            "Ridge probes measure linear decodability only; they do not establish regeneration or causality.",
        ],
        "split": {name: [int(x) for x in values.tolist()] for name, values in split.items()},
        "permutation_seeds": {
            "ssdc_rpi": int(permutation_seed),
            "probe_train_rpi": int(permutation_seed),
            "probe_eval_rpi": int(probe_eval_seed),
        },
        "ssdc": {
            "clean": {
                stream: {
                    "scores": [float(x) for x in clean_streams[stream]],
                    "summary": common.summarize_curve(clean_streams[stream]),
                }
                for stream in STREAM_NAMES
            },
            "rpi": {
                stream: {
                    "scores": [float(x) for x in rpi_streams[stream]],
                    "summary": common.summarize_curve(rpi_streams[stream]),
                }
                for stream in STREAM_NAMES
            },
            "delta": {
                "clean": {
                    "attention_residual_delta": [
                        float(x) for x in clean_streams["attention_residual_delta"]
                    ],
                    "mlp_residual_delta": [
                        float(x) for x in clean_streams["mlp_residual_delta"]
                    ],
                    "mlp_residual_delta_well_defined": bool(
                        clean_streams["mlp_residual_delta_well_defined"]
                    ),
                },
                "rpi": {
                    "attention_residual_delta": [
                        float(x) for x in rpi_streams["attention_residual_delta"]
                    ],
                    "mlp_residual_delta": [
                        float(x) for x in rpi_streams["mlp_residual_delta"]
                    ],
                    "mlp_residual_delta_well_defined": bool(
                        rpi_streams["mlp_residual_delta_well_defined"]
                    ),
                },
            },
        },
        "layers": layer_metrics,
    }


def run_experiment(
    model="both",
    number_images=1000,
    batch_size=128,
    sample_seed=0,
    permutation_seed=0,
    half=False,
    hf_token=None,
):
    kinds = ["ape", "rope"] if model == "both" else [model]
    results = {
        "metadata": {
            "experiment": "attention_output_analysis",
            "number_images": int(number_images),
            "batch_size": int(batch_size),
            "sample_seed": int(sample_seed),
            "permutation_seed": int(permutation_seed),
            "requested_half": bool(half),
        },
        "models": {},
    }

    for kind in kinds:
        dataset = common.load_imagenet(token=hf_token, shuffle=True, seed=sample_seed)
        results["models"][kind] = run_model_experiment(
            kind,
            dataset,
            number_images=number_images,
            batch_size=batch_size,
            sample_seed=sample_seed,
            permutation_seed=permutation_seed,
            half=half,
        )
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Attention-output SSDC and RPI ridge probes.")
    parser.add_argument("--model", choices=["ape", "rope", "both"], default="both")
    parser.add_argument("--number-images", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--permutation-seed", type=int, default=0)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--hf-token", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    results = run_experiment(
        model=args.model,
        number_images=args.number_images,
        batch_size=args.batch_size,
        sample_seed=args.sample_seed,
        permutation_seed=args.permutation_seed,
        half=args.half,
        hf_token=args.hf_token,
    )

    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    out_path = os.path.join(common.RESULTS_DIR, f"attention_output_analysis{suffix}.json")
    common.save_json(results, out_path)
    print(f"saved {out_path}")
    return results


if __name__ == "__main__":
    main()
