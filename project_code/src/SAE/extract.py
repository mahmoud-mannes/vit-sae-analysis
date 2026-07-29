"""Build an SAE activation store from a real ViT.

Streams images, runs the ViT, captures the residual stream entering a chosen
block, and returns the patch token activations as a single ``[n_tokens, d_model]``
tensor. Class / register tokens are dropped by default because their statistics
differ sharply from patch tokens and would otherwise soak up SAE capacity.

This reuses the repository's own model and block locators so the activations are
exactly the quantity the SSDC and ablation experiments read (the input to a
block, i.e. the post embedding residual stream). It works for both the
transformers APE model and the timm RoPE model.
"""

from __future__ import annotations

import os
import sys
import torch

_SRC = os.path.abspath(os.path.dirname(__file__) + "/..")
if _SRC not in sys.path:
    sys.path.append(_SRC)

from main.load_models import load_model, get_vit_blocks, num_prefix_tokens  # noqa: E402


@torch.no_grad()
def extract_residual_activations(
    kind: str = "ape",
    layer: int = 6,
    number_images: int = 640,
    batch_size: int = 32,
    drop_prefix: bool = True,
    token=None,
    dataset_id: str = "ILSVRC/imagenet-1k",
    fallback_id: str = "benjamin-paine/imagenet-1k-256x256",
    device: str | None = None,
    log=print,
):
    """Return ``(activations [n_tokens, d_model], meta dict)``."""
    from experiments.common import load_imagenet, get_hf_token

    token = get_hf_token(token)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, processor, source = load_model(kind, device=device)
    blocks = get_vit_blocks(model, source)
    if not (0 <= layer < len(blocks)):
        raise ValueError(f"layer {layer} out of range for {len(blocks)} blocks")
    n_prefix = num_prefix_tokens(model, source) if drop_prefix else 0

    captured = []

    def hook(module, inputs, output):
        captured.append(inputs[0].detach())

    handle = blocks[layer].register_forward_hook(hook)

    ds = load_imagenet(streaming=True, token=token, dataset_id=dataset_id, fallback_id=fallback_id)
    it = iter(ds)

    store_chunks = []
    seen = 0
    batch_imgs = []

    def run_batch(imgs):
        if source == "transformers":
            px = torch.cat(
                [processor(images=im, return_tensors="pt")["pixel_values"] for im in imgs], dim=0
            ).to(device)
            model(pixel_values=px)
        else:
            px = torch.stack([processor(im) for im in imgs], dim=0).to(device)
            model(px)
        act = captured.pop()  # [B, T, d_model]
        if n_prefix:
            act = act[:, n_prefix:, :]
        store_chunks.append(act.reshape(-1, act.shape[-1]).float().cpu())

    while seen < number_images:
        try:
            item = next(it)
        except StopIteration:
            break
        batch_imgs.append(item["image"].convert("RGB"))
        seen += 1
        if len(batch_imgs) == batch_size:
            run_batch(batch_imgs)
            batch_imgs = []
            if seen % (batch_size * 4) == 0:
                log(f"  extracted {seen}/{number_images} images", flush=True)
    if batch_imgs:
        run_batch(batch_imgs)

    handle.remove()
    acts = torch.cat(store_chunks, dim=0).contiguous()
    meta = {
        "kind": kind, "source": source, "layer": layer,
        "number_images": seen, "n_prefix_dropped": n_prefix,
        "n_tokens": acts.shape[0], "d_model": acts.shape[1],
    }
    log(f"  activation store: {tuple(acts.shape)} from {seen} images", flush=True)
    return acts, meta
