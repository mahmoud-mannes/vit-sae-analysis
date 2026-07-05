"""Spatial Similarity Distance Correlation (SSDC).

SSDC measures how strongly representational similarity between two patch tokens
tracks their spatial closeness. For a layer we take the token representations,
build the T by T cosine similarity matrix S averaged over the image batch, build
the spatial distance matrix D from the tokens' grid coordinates, and report the
Spearman rank correlation between similarity and negative distance over all token
pairs:

    SSDC = spearman( { S_ij }_{i<j}, { -D_ij }_{i<j} ).

A high SSDC means spatially near tokens are represented similarly. Under Random
Permutation at Inference (RPI) the patch contents are shuffled while the
positional signal stays pinned to the sequence index, so SSDC that survives the
shuffle reflects structure anchored to token position rather than to content.

This file keeps the pure metric (spatial_similarity_distance_correlation) and a
model level evaluator (evaluate_ssdc). The evaluator captures the post norm
residual entering each block by hooking the attention submodule input, which is
the same quantity for both the transformers and timm ViTs.
"""

import os
import sys

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
import torch

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
from main.prep_data import prep_data
from main.model import predict


def spatial_similarity_distance_correlation(S, grid_size, metric="manhattan"):
    """Spearman correlation between token similarity and negative spatial distance.

    S : (T, T) similarity matrix over patch tokens laid out in row major order.
    grid_size : side length of the square token grid, so grid_size**2 == T.
    metric : "manhattan" (L1, as in the reference runs) or "euclidean".
    """
    grid_size = int(round(grid_size))
    T = S.shape[0]
    assert S.shape[0] == S.shape[1], "Similarity matrix must be square (T by T)"
    assert grid_size * grid_size == T, "grid_size**2 must equal T"

    # Spatial coordinates in row major order: token k sits at (k // G, k % G).
    coords = np.stack(
        np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)

    if metric == "manhattan":
        dists = cdist(coords, coords, metric="cityblock")
    elif metric == "euclidean":
        dists = cdist(coords, coords, metric="euclidean")
    else:
        raise ValueError("metric must be 'manhattan' or 'euclidean'")

    iu = np.triu_indices(T, k=1)  # unique unordered pairs, excludes the diagonal
    sim_vals = S[iu]
    dist_vals = dists[iu]

    corr, _ = spearmanr(-dist_vals, sim_vals)
    return corr


def _make_accumulating_hook(store, layer_idx):
    """Return a forward hook that folds this batch's per token cosine similarity
    into a running sum, so we never hold every image's activations at once.

    The hook reads inputs[0], the tensor fed into the attention submodule, which
    is the post norm residual stream entering the block. Summing the per image
    similarity matrices and dividing by the image count at the end gives the
    batch averaged S used by SSDC.
    """

    def hook(module, inputs, output):
        tok = inputs[0].detach().float()  # [B, T, C]
        norm = tok / (tok.norm(dim=-1, keepdim=True) + 1e-8)
        cos = norm @ norm.transpose(-2, -1)  # [B, T, T]
        summed = cos.sum(0).cpu()  # [T, T]
        if layer_idx in store["sum"]:
            store["sum"][layer_idx] += summed
            store["count"][layer_idx] += tok.shape[0]
        else:
            store["sum"][layer_idx] = summed
            store["count"][layer_idx] = tok.shape[0]

    return hook


def _register_block_hooks(model, source, store):
    """Hook every block's attention submodule input and return the handles."""
    handles = []
    if source == "transformers":
        blocks = model.vit.encoder.layer
        for i, blk in enumerate(blocks):
            handles.append(blk.attention.register_forward_hook(_make_accumulating_hook(store, i)))
    elif source == "timm":
        blocks = model.blocks
        for i, blk in enumerate(blocks):
            handles.append(blk.attn.register_forward_hook(_make_accumulating_hook(store, i)))
    else:
        raise ValueError("source must be 'timm' or 'transformers'")
    return handles


def evaluate_ssdc(
    model,
    processor,
    dataset,
    source,
    RPI=False,
    magnitude=1.0,
    number_images=1000,
    batch_size=256,
    metric="manhattan",
    n_prefix=None,
):
    """Compute per layer SSDC for a model over a streamed image sample.

    Returns
    -------
    ssdc_scores : list of float, one SSDC per block (length == number of blocks).
    cosine_maps : list of (T, T) numpy arrays, the batch averaged similarity
        matrices including the prefix (class) token, useful for visualisation.

    Notes
    -----
    - RPI and magnitude are forwarded to predict, which installs the random
      permutation hook and the positional scaling for those interventions.
    - number_images caps how much of the (streaming) split is consumed. The
      reference curves were produced from a sample on this order; larger samples
      give smoother estimates at higher cost.
    """
    if source not in ("timm", "transformers"):
        raise ValueError("source must be 'timm' or 'transformers'")

    if n_prefix is None:
        n_prefix = 1 if source == "transformers" else int(getattr(model, "num_prefix_tokens", 1))

    dataloader = prep_data(
        dataset, processor, source, number_images=number_images, batch_size=batch_size
    )

    store = {"sum": {}, "count": {}}
    handles = _register_block_hooks(model, source, store)
    try:
        predict(model, dataloader, source, RPI, magnitude)
    finally:
        for handle in handles:
            handle.remove()

    ssdc_scores = []
    cosine_maps = []
    for i in sorted(store["sum"].keys()):
        mean_cos = (store["sum"][i] / store["count"][i]).numpy()
        cosine_maps.append(mean_cos)

        patch_cos = mean_cos[n_prefix:, n_prefix:]  # drop the class token(s)
        grid_size = int(round(patch_cos.shape[0] ** 0.5))
        ssdc = spatial_similarity_distance_correlation(patch_cos, grid_size=grid_size, metric=metric)
        ssdc_scores.append(float(ssdc))

    return ssdc_scores, cosine_maps
