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
from main.load_models import get_vit_blocks, get_block_attention


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


PAIRWISE_SIMILARITY_METRICS = ("cosine", "centered_cosine", "negative_euclidean")


def _resolve_pairwise_metrics(metrics):
    if metrics is None:
        metrics = PAIRWISE_SIMILARITY_METRICS
    elif isinstance(metrics, str):
        metrics = (metrics,)
    else:
        metrics = tuple(metrics)

    ordered = []
    seen = set()
    for name in metrics:
        if name not in PAIRWISE_SIMILARITY_METRICS:
            raise ValueError(
                f"metric {name!r} must be one of {PAIRWISE_SIMILARITY_METRICS}"
            )
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return tuple(ordered)


def pairwise_similarity_matrices(tokens, metrics=None):
    """Return per-image pairwise matrices where larger means more similar."""
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [B, T, C]")

    resolved = _resolve_pairwise_metrics(metrics)
    tokens = tokens.float()
    outputs = {}

    if "cosine" in resolved:
        normalized = tokens / tokens.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        outputs["cosine"] = normalized @ normalized.transpose(-2, -1)

    if "centered_cosine" in resolved:
        centered = tokens - tokens.mean(dim=1, keepdim=True)
        centered = centered / centered.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        outputs["centered_cosine"] = centered @ centered.transpose(-2, -1)

    if "negative_euclidean" in resolved:
        outputs["negative_euclidean"] = -torch.cdist(tokens, tokens, p=2)

    return outputs


class SSDCAccumulator:
    """Accumulate average pairwise token-similarity matrices by arbitrary key."""

    def __init__(self, metrics=("cosine",), n_prefix=0, spatial_metric="manhattan"):
        self.metrics = _resolve_pairwise_metrics(metrics)
        self.n_prefix = int(n_prefix)
        self.spatial_metric = spatial_metric
        self._store = {}

    def _entry(self, key):
        if key not in self._store:
            self._store[key] = {name: None for name in self.metrics}
            self._store[key]["count"] = 0
        return self._store[key]

    def _resolve_metric(self, metric):
        resolved = _resolve_pairwise_metrics(metric)
        if len(resolved) != 1:
            raise ValueError("exactly one pairwise metric must be requested")
        name = resolved[0]
        if name not in self.metrics:
            raise ValueError(
                f"metric {name!r} was not accumulated; available metrics are {self.metrics}"
            )
        return name

    def add(self, key, tokens):
        matrices = pairwise_similarity_matrices(tokens, metrics=self.metrics)
        entry = self._entry(key)
        for name, matrix in matrices.items():
            summed = matrix.detach().sum(dim=0).cpu()
            entry[name] = summed if entry[name] is None else entry[name] + summed
        entry["count"] += int(tokens.shape[0])

    def keys(self):
        return self._store.keys()

    def count(self, key):
        return self._store[key]["count"]

    def mean_pairwise_matrix(self, key, metric="cosine", remove_prefix=False):
        name = self._resolve_metric(metric)
        entry = self._store[key]
        mean_matrix = (entry[name] / entry["count"]).numpy()
        if remove_prefix:
            mean_matrix = mean_matrix[self.n_prefix :, self.n_prefix :]
        return mean_matrix

    def mean_pairwise_matrices(self, metric="cosine", remove_prefix=False):
        return {
            key: self.mean_pairwise_matrix(key, metric=metric, remove_prefix=remove_prefix)
            for key in self._store
        }

    def ssdc(self, key, metric="cosine", remove_prefix=True):
        mean_matrix = self.mean_pairwise_matrix(
            key, metric=metric, remove_prefix=remove_prefix
        )
        grid_size = int(round(mean_matrix.shape[0] ** 0.5))
        return float(
            spatial_similarity_distance_correlation(
                mean_matrix, grid_size=grid_size, metric=self.spatial_metric
            )
        )

    def ssdc_by_key(self, metric="cosine", remove_prefix=True):
        return {
            key: self.ssdc(key, metric=metric, remove_prefix=remove_prefix)
            for key in self._store
        }


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
        summed = pairwise_similarity_matrices(tok, metrics=("cosine",))["cosine"].sum(0).cpu()
        if layer_idx in store["sum"]:
            store["sum"][layer_idx] += summed
            store["count"][layer_idx] += tok.shape[0]
        else:
            store["sum"][layer_idx] = summed
            store["count"][layer_idx] = tok.shape[0]

    return hook


def _register_block_hooks(model, source, store):
    """Hook every block's attention submodule input and return the handles.

    The input to the attention submodule is the post norm residual entering the
    block, the same quantity for both models and both transformers layouts.
    """
    handles = []
    for i, blk in enumerate(get_vit_blocks(model, source)):
        attn = get_block_attention(blk, source)
        handles.append(attn.register_forward_hook(_make_accumulating_hook(store, i)))
    return handles


def _make_ssdc_accumulator_hook(accumulator, key):
    def hook(module, inputs, output):
        accumulator.add(key, inputs[0].detach())

    return hook


def _register_ssdc_accumulator_hooks(model, source, accumulator):
    handles = []
    for i, blk in enumerate(get_vit_blocks(model, source)):
        attn = get_block_attention(blk, source)
        handles.append(attn.register_forward_hook(_make_ssdc_accumulator_hook(accumulator, i)))
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
    half=True,
    num_workers=None,
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
        dataset, processor, source, number_images=number_images,
        batch_size=batch_size, half=half, num_workers=num_workers,
    )

    accumulator = SSDCAccumulator(metrics=("cosine",), n_prefix=n_prefix, spatial_metric=metric)
    handles = _register_ssdc_accumulator_hooks(model, source, accumulator)
    try:
        predict(model, dataloader, source, RPI, magnitude, half=half)
    finally:
        for handle in handles:
            handle.remove()

    ssdc_scores = []
    cosine_maps = []
    for i in sorted(accumulator.keys()):
        mean_cos = accumulator.mean_pairwise_matrix(i, metric="cosine", remove_prefix=False)
        cosine_maps.append(mean_cos)
        ssdc_scores.append(accumulator.ssdc(i, metric="cosine", remove_prefix=True))

    return ssdc_scores, cosine_maps
