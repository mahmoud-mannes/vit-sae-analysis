"""Effective rank of the residual stream, and its tie to rank collapse.

Dong, Cordonnier and Loukas (2021), "Attention is not all you need: pure
attention loses rank doubly exponentially with depth", show that stacked self
attention without MLPs or skip connections drives token representations toward
rank one, where every token collapses to the same vector. MLP sublayers and skip
connections counteract that collapse. This is the mechanistic reason to expect
MLP ablation to damage any structure carried by the residual stream, including
the index anchored spatial structure that SSDC measures. This module lets us
watch representational rank next to SSDC under the same ablations.

Effective rank (Roy and Vetterli, 2007): given singular values s_1..s_r of the
token matrix, set p_i = s_i / sum_j s_j and report exp(-sum_i p_i log p_i). It is
1 when the matrix is rank one (full collapse) and grows toward the embedding
dimension as energy spreads across directions.
"""

import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
from main.prep_data import prep_data
from main.model import predict


def batched_effective_rank(tok, center=False, eps=1e-12):
    """Effective rank of each image's token matrix.

    tok : [B, T, C] tensor (the per image token representations at one layer).
    center : subtract the mean token first, which isolates the "collapse toward a
        common vector" component that Dong et al. describe.
    Returns a [B] tensor of effective ranks.
    """
    tok = tok.detach().float()
    if center:
        tok = tok - tok.mean(dim=1, keepdim=True)
    s = torch.linalg.svdvals(tok).clamp_min(0.0)  # [B, K]
    total = s.sum(dim=1, keepdim=True) + eps
    p = s / total
    entropy = -(p * torch.log(p + eps)).sum(dim=1)  # [B]
    return torch.exp(entropy)


def effective_rank_from_matrix(X, center=False, eps=1e-12):
    """Effective rank of a single [T, C] matrix (numpy or torch)."""
    if not torch.is_tensor(X):
        X = torch.as_tensor(np.asarray(X))
    return float(batched_effective_rank(X.unsqueeze(0), center=center, eps=eps)[0])


def _make_rank_hook(store, layer_idx, center):
    def hook(module, inputs, output):
        tok = inputs[0]  # [B, T, C], the post norm residual entering the block
        ranks = batched_effective_rank(tok, center=center)
        summed = float(ranks.sum().item())
        n = tok.shape[0]
        if layer_idx in store["sum"]:
            store["sum"][layer_idx] += summed
            store["count"][layer_idx] += n
        else:
            store["sum"][layer_idx] = summed
            store["count"][layer_idx] = n

    return hook


def _register_block_hooks(model, source, store, center):
    handles = []
    if source == "transformers":
        for i, blk in enumerate(model.vit.encoder.layer):
            handles.append(blk.attention.register_forward_hook(_make_rank_hook(store, i, center)))
    elif source == "timm":
        for i, blk in enumerate(model.blocks):
            handles.append(blk.attn.register_forward_hook(_make_rank_hook(store, i, center)))
    else:
        raise ValueError("source must be 'timm' or 'transformers'")
    return handles


def evaluate_effective_rank(
    model,
    processor,
    dataset,
    source,
    RPI=False,
    magnitude=1.0,
    number_images=1000,
    batch_size=256,
    center=False,
):
    """Per layer effective rank over a streamed image sample.

    Composes with AblationController exactly like evaluate_ssdc, so the same
    layer windowed MLP / attention ablations can be measured through the rank
    lens as well as the SSDC lens.
    """
    if source not in ("timm", "transformers"):
        raise ValueError("source must be 'timm' or 'transformers'")

    dataloader = prep_data(
        dataset, processor, source, number_images=number_images, batch_size=batch_size
    )

    store = {"sum": {}, "count": {}}
    handles = _register_block_hooks(model, source, store, center)
    try:
        predict(model, dataloader, source, RPI, magnitude)
    finally:
        for handle in handles:
            handle.remove()

    return [store["sum"][i] / store["count"][i] for i in sorted(store["sum"].keys())]
