"""Layer-wise activation patching for positional grounding.

This experiment compares two runs of the same images:

  clean: normal patch order
  rpi  : patch tokens randomly permuted before positional information is applied

For a chosen layer/component, we cache the clean activation and patch it into the
RPI run for the same batch. The readouts are top-1 accuracy and the SSDC curve of
the patched run. If a component carries positional grounding, patching it should
move the RPI readouts back toward the clean baseline.

The implementation covers the HuggingFace APE ViT
(`google/vit-base-patch16-224`) and the timm RoPE ViT
(`vit_base_patch16_rope_224.naver_in1k`). It deliberately keeps memory low by
recomputing the clean activation per batch/condition instead of caching every
activation for the whole dataset.
"""

import argparse
import os
import sys
import types
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
from metrics.ssdc import SSDCAccumulator


def _replace_tensor(output, value):
    if isinstance(output, tuple):
        return (value,) + tuple(output[1:])
    return value


def _clone_cpu(x):
    return x.detach().cpu().clone()


def _to_like(x, ref):
    return x.to(device=ref.device, dtype=ref.dtype)


def resolve_patch_target(model, source, layer, component, stream=None):
    blocks = get_vit_blocks(model, source)
    block = blocks[layer]
    if component == "residual":
        return block, "residual"
    if component == "attn":
        return get_block_attention(block, source), "attn"
    if component == "mlp":
        module, mode = get_block_mlp(block, source)
        return module, f"mlp_{mode}"
    if component == "head":
        if source == "timm":
            return block.attn.proj, "head_pre_project"
        if hasattr(block.attention, "attention"):
            return block.attention.attention, "head_context"
        if hasattr(block.attention, "o_proj"):
            return block.attention.o_proj, "head_pre_project"
        raise AttributeError("could not locate the attention head context tensor")
    if component == "qkv":
        if stream not in {"query", "key", "value"}:
            raise ValueError("qkv component requires stream in {'query', 'key', 'value'}")
        if source == "timm":
            if hasattr(block.attn, "qkv") and block.attn.qkv is not None:
                return block.attn, "timm_rope_stream"
            proj_name = {"query": "q_proj", "key": "k_proj", "value": "v_proj"}[stream]
            if hasattr(block.attn, proj_name):
                return getattr(block.attn, proj_name), "qkv"
            raise AttributeError(f"could not locate {stream} projection")
        if hasattr(block.attention, "attention"):
            return getattr(block.attention.attention, stream), "qkv"
        proj_name = {"query": "q_proj", "key": "k_proj", "value": "v_proj"}[stream]
        if hasattr(block.attention, proj_name):
            return getattr(block.attention, proj_name), "qkv"
        raise AttributeError(f"could not locate {stream} projection")
    raise ValueError(f"unknown component {component!r}")


def num_attention_heads(model, source, layer):
    block = get_vit_blocks(model, source)[layer]
    if source == "timm":
        return int(block.attn.num_heads)
    attn = getattr(block.attention, "attention", block.attention)
    if hasattr(attn, "num_attention_heads"):
        return int(attn.num_attention_heads)
    if hasattr(attn, "num_heads"):
        return int(attn.num_heads)
    return int(model.config.num_attention_heads)


class _ForwardPatchHandle:
    def __init__(self, module, original_forward):
        self.module = module
        self.original_forward = original_forward

    def remove(self):
        self.module.forward = self.original_forward


def _install_timm_rope_stream_forward(module, spec, cache=None, patch=False):
    """Patch timm AttentionRope internals after rotary embedding.

    The qkv Linear output is pre-RoPE. For RoPE, the causally relevant query/key
    streams are the post-rotation tensors inside attention, so this wrapper
    mirrors timm's AttentionRope.forward and intercepts q/k after `apply_rot`.
    """
    original_forward = module.forward
    globals_ = original_forward.__globals__
    apply_rot_embed_cat = globals_["apply_rot_embed_cat"]
    resolve_self_attn_mask = globals_["resolve_self_attn_mask"]
    maybe_add_mask = globals_["maybe_add_mask"]
    stream = spec["stream"]

    def forward(self, x, rope=None, attn_mask=None, is_causal=False):
        B, N, C = x.shape

        if self.qkv is not None:
            qkv = self.qkv(x)
            qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
        else:
            q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            npt = self.num_prefix_tokens
            half = getattr(self, "rotate_half", False)
            q = torch.cat([q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope, half=half)], dim=2).type_as(v)
            k = torch.cat([k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope, half=half)], dim=2).type_as(v)

        if cache is not None and not patch:
            value = {"query": q, "key": k, "value": v}[stream]
            cache["value"] = _clone_cpu(value)
        if patch:
            clean = _to_like(cache["value"], q)
            if stream == "query":
                q = clean
            elif stream == "key":
                k = clean
            elif stream == "value":
                v = clean
            else:
                raise ValueError(f"unknown stream {stream!r}")

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn_bias = resolve_self_attn_mask(N, attn, attn_mask, is_causal)
        attn = maybe_add_mask(attn, attn_bias)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v

        out = out.transpose(1, 2).reshape(B, N, self.attn_dim)
        out = self.norm(out)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    module.forward = types.MethodType(forward, module)
    return _ForwardPatchHandle(module, original_forward)


def install_clean_cache_hook(model, source, spec, cache):
    """Cache one clean component activation for a paired patching pass."""
    module, mode = resolve_patch_target(
        model, source, spec["layer"], spec["component"], spec.get("stream")
    )

    def hook(module, inputs, output):
        if mode == "mlp_residual":
            value = common.as_tensor(output) - inputs[1]
        elif mode in ("head_context", "head_pre_project"):
            context = inputs[0] if mode == "head_pre_project" else common.as_tensor(output)
            n_heads = spec["n_heads"]
            head_dim = context.shape[-1] // n_heads
            heads = context.reshape(context.shape[0], context.shape[1], n_heads, head_dim)
            value = heads[:, :, spec["head"], :]
        else:
            value = common.as_tensor(output)
        cache["value"] = _clone_cpu(value)

    if mode == "timm_rope_stream":
        return _install_timm_rope_stream_forward(module, spec, cache=cache, patch=False)
    if mode == "head_pre_project":
        def pre_hook(module, inputs):
            context = inputs[0]
            n_heads = spec["n_heads"]
            head_dim = context.shape[-1] // n_heads
            heads = context.reshape(context.shape[0], context.shape[1], n_heads, head_dim)
            cache["value"] = _clone_cpu(heads[:, :, spec["head"], :])
            return None

        return module.register_forward_pre_hook(pre_hook)
    return module.register_forward_hook(hook)


def install_patch_hook(model, source, spec, cache):
    """Replace one component output with a cached clean activation."""
    module, mode = resolve_patch_target(
        model, source, spec["layer"], spec["component"], spec.get("stream")
    )

    def hook(module, inputs, output):
        clean = cache["value"]
        if mode == "mlp_residual":
            return inputs[1] + _to_like(clean, inputs[1])
        if mode == "head_context":
            context = common.as_tensor(output)
            n_heads = spec["n_heads"]
            head_dim = context.shape[-1] // n_heads
            heads = context.reshape(context.shape[0], context.shape[1], n_heads, head_dim).clone()
            heads[:, :, spec["head"], :] = _to_like(clean, heads)
            patched = heads.reshape_as(context)
            return _replace_tensor(output, patched)
        patched = _to_like(clean, common.as_tensor(output))
        return _replace_tensor(output, patched)

    if mode == "timm_rope_stream":
        return _install_timm_rope_stream_forward(module, spec, cache=cache, patch=True)
    if mode == "head_pre_project":
        def pre_hook(module, inputs):
            context = inputs[0]
            n_heads = spec["n_heads"]
            head_dim = context.shape[-1] // n_heads
            heads = context.reshape(context.shape[0], context.shape[1], n_heads, head_dim).clone()
            heads[:, :, spec["head"], :] = _to_like(cache["value"], heads)
            patched = heads.reshape_as(context)
            return (patched,) + tuple(inputs[1:])

        return module.register_forward_pre_hook(pre_hook)
    return module.register_forward_hook(hook)


def install_block_output_ssdc_hooks(model, source, accumulator):
    """Capture post-block residual streams with the shared SSDC accumulator."""
    handles = []
    for layer, block in enumerate(get_vit_blocks(model, source)):
        def make_hook(idx):
            def hook(module, inputs, output):
                accumulator.add(idx, common.as_tensor(output).detach())
            return hook

        handles.append(block.register_forward_hook(make_hook(layer)))
    return handles


def block_output_ssdc_scores(accumulator):
    return [
        accumulator.ssdc(layer, metric="cosine", remove_prefix=True)
        for layer in sorted(accumulator.keys())
    ]


def run_baseline(model, source, batches, perm=None, half=True):
    accumulator = SSDCAccumulator(n_prefix=num_prefix_tokens(model, source))
    correct = total = 0
    handles = install_block_output_ssdc_hooks(model, source, accumulator)
    with torch.inference_mode(), ExitStack() as stack:
        for handle in handles:
            stack.callback(handle.remove)
        if perm is not None:
            stack.callback(common.install_fixed_rpi_hook(model, source, perm).remove)
        for pixel_values, labels in batches:
            pixel_values = common.prepare_pixel_values(pixel_values, model, half=half)
            logits = common.forward_logits(model, source, pixel_values)
            c, t = common.accuracy_counts(logits, labels)
            correct += c
            total += t
    scores = block_output_ssdc_scores(accumulator)
    out = common.summarize_ssdc(scores)
    out["accuracy"] = correct / total
    out["n_images"] = total
    return out


def run_patch_condition(model, source, batches, perm, spec, half=True):
    accumulator = SSDCAccumulator(n_prefix=num_prefix_tokens(model, source))
    correct = total = 0
    with torch.inference_mode():
        for pixel_values, labels in batches:
            pixel_values = common.prepare_pixel_values(pixel_values, model, half=half)

            # The cache pass must not contribute to SSDC. Metric hooks are
            # installed only for the RPI + patch pass below.
            cache = {}
            clean_handle = install_clean_cache_hook(model, source, spec, cache)
            try:
                common.forward_logits(model, source, pixel_values)
            finally:
                clean_handle.remove()

            with ExitStack() as stack:
                stack.callback(common.install_fixed_rpi_hook(model, source, perm).remove)

                # Register the patch before block-level SSDC hooks. For a
                # residual-stream patch this ensures the patched block output,
                # rather than the original output, is what SSDC observes.
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
    out = common.summarize_ssdc(scores)
    out["accuracy"] = correct / total
    out["n_images"] = total
    return out


def build_specs(
    model,
    source,
    components,
    include_heads=False,
    include_qkv=False,
    layers=None,
    heads=None,
):
    n_layers = len(get_vit_blocks(model, source))
    layer_ids = common.validate_layer_indices(
        layers, n_layers, name="layers", default=range(n_layers)
    )
    unknown_components = set(components) - {"residual", "attn", "mlp"}
    if unknown_components:
        raise ValueError(f"unknown components: {sorted(unknown_components)}")
    specs = []
    for component in components:
        for layer in layer_ids:
            specs.append(
                {
                    "name": f"{component}_L{layer:02d}",
                    "component": component,
                    "layer": layer,
                }
            )
    if include_heads:
        for layer in layer_ids:
            n_heads = num_attention_heads(model, source, layer)
            head_ids = list(range(n_heads)) if heads is None else [int(x) for x in heads]
            for head in head_ids:
                if head < 0 or head >= n_heads:
                    raise ValueError(
                        f"head {head} is out of range for layer {layer} "
                        f"with {n_heads} heads"
                    )
                specs.append(
                    {
                        "name": f"head_L{layer:02d}_H{head:02d}",
                        "component": "head",
                        "layer": layer,
                        "head": head,
                        "n_heads": n_heads,
                    }
                )
    if include_qkv:
        n_heads_by_layer = {
            layer: num_attention_heads(model, source, layer) for layer in layer_ids
        }
        for stream in ("query", "key", "value"):
            for layer in layer_ids:
                specs.append(
                    {
                        "name": f"{stream}_L{layer:02d}",
                        "component": "qkv",
                        "stream": stream,
                        "layer": layer,
                        "n_heads": n_heads_by_layer[layer],
                    }
                )
    return specs


def run_experiment(
    model_kind="ape",
    number_images=128,
    batch_size=32,
    components=("residual", "attn", "mlp"),
    include_heads=False,
    include_qkv=False,
    layers=None,
    heads=None,
    hf_token=None,
    dataset_id="ILSVRC/imagenet-1k",
    fallback_id="benjamin-paine/imagenet-1k-256x256",
    corruption_type=None,
    severity=5,
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
    batches = common.collect_batches(
        dataset, processor, source, number_images, batch_size,
        corruption_type=corruption_type, severity=severity,
    )
    if not batches:
        raise RuntimeError("no batches collected from dataset")

    first = batches[0][0]
    n_patches = common.patch_token_count(model, source, first, half=half)
    perm = common.seeded_patch_permutation(n_patches, permutation_seed)

    results = {
        "model": model_kind,
        "source": source,
        "number_images": int(sum(labels.numel() for _, labels in batches)),
        "batch_size": batch_size,
        "components": list(components),
        "include_heads": include_heads,
        "include_qkv": include_qkv,
        "layers": None if layers is None else [int(x) for x in layers],
        "heads": None if heads is None else [int(x) for x in heads],
        "corruption_type": corruption_type,
        "severity": severity,
        "shuffle": shuffle,
        "sample_seed": sample_seed,
        "buffer_size": buffer_size,
        "permutation_seed": permutation_seed,
        "baselines": {},
        "patches": {},
    }

    print("== baseline: clean ==")
    results["baselines"]["clean"] = run_baseline(model, source, batches, perm=None, half=half)
    print(results["baselines"]["clean"])

    print("== baseline: rpi ==")
    results["baselines"]["rpi"] = run_baseline(model, source, batches, perm=perm, half=half)
    print(results["baselines"]["rpi"])

    for spec in build_specs(model, source, components, include_heads, include_qkv, layers, heads):
        print(f"== patch: {spec['name']} ==")
        result = run_patch_condition(model, source, batches, perm, spec, half=half)
        result["spec"] = {k: v for k, v in spec.items() if k != "n_heads"}
        results["patches"][spec["name"]] = result
        print(
            f"acc={result['accuracy']:.3f} final={result['final']:.3f} "
            f"peak={result['peak']:.3f} auc={result['auc']:.3f}"
        )

    suffix = f"_{output_suffix}" if output_suffix else ""
    out_path = os.path.join(common.RESULTS_DIR, f"activation_patching_{model_kind}{suffix}.json")
    common.save_json(results, out_path)
    print(f"saved {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Layer-wise activation patching under RPI.")
    parser.add_argument("--model", choices=["ape", "rope"], default="ape")
    parser.add_argument("--number-images", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--components",
        nargs="+",
        choices=["residual", "attn", "mlp"],
        default=["residual", "attn", "mlp"],
    )
    parser.add_argument("--include-heads", action="store_true")
    parser.add_argument("--include-qkv", action="store_true")
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--heads", type=int, nargs="+", default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--dataset-id", default="ILSVRC/imagenet-1k")
    parser.add_argument("--fallback-id", default="benjamin-paine/imagenet-1k-256x256")
    parser.add_argument("--corruption", default=None)
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=2000)
    parser.add_argument("--permutation-seed", type=int, default=0)
    parser.add_argument("--no-half", action="store_true")
    parser.add_argument("--output-suffix", default=None)
    args = parser.parse_args()

    run_experiment(
        model_kind=args.model,
        number_images=args.number_images,
        batch_size=args.batch_size,
        components=tuple(args.components),
        include_heads=args.include_heads,
        include_qkv=args.include_qkv,
        layers=args.layers,
        heads=args.heads,
        hf_token=args.hf_token,
        dataset_id=args.dataset_id,
        fallback_id=args.fallback_id,
        corruption_type=args.corruption,
        severity=args.severity,
        shuffle=args.shuffle,
        sample_seed=args.sample_seed,
        buffer_size=args.buffer_size,
        permutation_seed=args.permutation_seed,
        half=not args.no_half,
        output_suffix=args.output_suffix,
    )
if __name__ == "__main__":
    main()
