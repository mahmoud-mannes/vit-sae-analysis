import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist
import sys
import torch
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
from main.prep_data import prep_data
from main.model import predict


def spatial_similarity_distance_correlation(S,grid_size, metric):

    T = S.shape[0]
    assert S.shape[0] == S.shape[1], "Similarity matrix must be square (T×T)"
    assert grid_size * grid_size == T, "grid_size^2 must equal T"

    # Spatial coordinates (row-major)
    coords = np.stack(
        np.meshgrid(
            np.arange(grid_size),
            np.arange(grid_size),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 2)  # (T, 2)

    # Pairwise spatial distances
    if metric == "manhattan":
        dists = cdist(coords, coords, metric="cityblock")
    elif metric == "euclidean":
        dists = cdist(coords, coords, metric="euclidean")
    else:
        raise ValueError("metric must be 'manhattan' or 'euclidean'")

    # Upper triangle (exclude diagonal)
    iu = np.triu_indices(T, k=1)

    sim_vals = S[iu]
    dist_vals = dists[iu]

    corr, _ = spearmanr(-dist_vals, sim_vals)
    return corr

def evaluate_ssdc(model, dataset, RPI = False, magnitude = 1.0):
    dataloader = prep_data(dataset)  

    token_inputs = {}

    def token_hook(name):
        def hook(module, inputs, output):
            token_inputs[name] = inputs[0].detach().cpu()
        return hook

    handles = []

    for name, module in model.named_modules():

        if name.startswith("vit.layers") and name.endswith("attention"): #encoder_layers.x.1 is the MultiHeadedAttention component of the encoder blocks
            handles.append(
                module.register_forward_hook(
                    token_hook(name)
                )
            )

    predict(model,dataloader, RPI, magnitude)

    for handle in handles:
        handle.remove()

    layer_names = sorted(
        token_inputs.keys(),
        key=lambda x: int(x.split(".")[1])
    )

    ssdc_scores = []
    cosine_maps = []

    for layer_name in layer_names:

        tok_inp = token_inputs[layer_name]

        norm_tok = tok_inp / (
            tok_inp.norm(dim=-1, keepdim=True) + 1e-8
        )

        cos_sim = (
            norm_tok
            @ norm_tok.transpose(-2, -1)
        )

        mean_cos = cos_sim.mean(0).numpy()

        cosine_maps.append(mean_cos)

        grid_size = (len(mean_cos) - 1) ** 0.5

        ssdc = spatial_similarity_distance_correlation(
            mean_cos[1:, 1:], # Exclude cls token,
            grid_size = grid_size,
            metric="manhattan"
        )

        ssdc_scores.append(ssdc.item())

    return ssdc_scores, cosine_maps