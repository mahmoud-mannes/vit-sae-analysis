import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from SAE_feature_analysis.top_candidates import get_top_candidates
from SAE_feature_analysis.visualize_feature_activation import (
    group_activations_by_position,
    mean_feature_activation_by_position,
)
import numpy as np
import torch

eps = 1e-6

def row_selectivity(matrix: torch.Tensor) -> float:
    """matrix: [grid, grid] mean activation map for one feature."""
    max_mean_row = matrix.mean(dim=1).max()
    matrix_mean = matrix.mean()
    return (max_mean_row / (matrix_mean + eps)).item()

def column_selectivity(matrix: torch.Tensor) -> float:
    return row_selectivity(matrix.T)

def top_selective_features(
    latent_activations: torch.Tensor,
    num_tokens: int = 197,
    num_prefix_tokens: int = 1,
    k_candidates: int = 3,
    verbose: bool = False,
):
    """
    Memory-efficient version.
    Precomputes position-grouped activations once, then only works
    with the small set of candidate features.
    """
    n_patch = num_tokens - num_prefix_tokens
    grid_size = int(n_patch ** 0.5)
    assert grid_size * grid_size == n_patch, "num_tokens - prefix must be a square"

    # ---------- 1. Group once ----------
    # Expected shape after grouping depends on your helper,
    # but we only need mean activation per (feature, position).
    grouped = group_activations_by_position(latent_activations)

    # ---------- 2. Precompute mean maps for features we will need ----------
    # First collect the union of all candidate features across positions
    candidate_sets = []
    for i in range(n_patch):
        top = get_top_candidates(
            latent_activations,
            target_position=i + num_prefix_tokens,
            k=k_candidates,
        )
        candidate_sets.append(top.indices.tolist())

    unique_features = sorted(set(f for subset in candidate_sets for f in subset))

    # mean_maps[feature] = [grid_size, grid_size]
    mean_maps = {}
    for f in unique_features:
        feature_list = mean_feature_activation_by_position(grouped, feature=f)
        # drop prefix, reshape to grid
        arr = np.asarray(feature_list[num_prefix_tokens:], dtype=np.float32)
        mean_maps[f] = torch.from_numpy(arr.reshape(grid_size, grid_size))

    # ---------- 3. Score only the precomputed maps ----------
    TSFPD = {}
    for i, candidates in enumerate(candidate_sets):
        row = i // grid_size
        col = i % grid_size

        if verbose:
            print(f"----- ROW {row} COLUMN {col} -----")

        best_col_score, best_col_feat = -1.0, None
        best_row_score, best_row_feat = -1.0, None

        for f in candidates:
            MFAP = mean_maps[f]
            fcs = column_selectivity(MFAP)
            frs = row_selectivity(MFAP)

            if fcs > best_col_score:
                best_col_score, best_col_feat = fcs, f
            if frs > best_row_score:
                best_row_score, best_row_feat = frs, f

        TSFPD[(row, col)] = (best_row_feat, best_col_feat)

        if verbose:
            print(f"Feature {best_col_feat}, Column selectivity {best_col_score:.3f}")
            print(f"Feature {best_row_feat}, Row selectivity {best_row_score:.3f}")

    return TSFPD