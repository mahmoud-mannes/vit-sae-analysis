from random import randint
import sys
import os

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from SAE_causal.feature_ablation_hook import attach_feature_ablation_hook

def ablate_features(
    model: nn.Module,
    source: str,
    model_type: str,
    SAE: nn.Module,
    layer: int,
    block: str,
    features_to_remove: list = None,
    top_features: bool = True,
    selectivity_scores_path: str = None,
    random_features: bool = False,
    k: int = 0
) -> RemovableHandle:
    """
    Attaches a feature ablation hook to the specified model and handles random baselines + top positional feature candidate ablation.
    For the residual stream, the feature ablation will happen before the block's forward pass. For the attention and MLP blocks, the feature ablation will happen after the block's forward pass.
    This is mainly because our SAE work on attention focuses on the attention output, since that's what'll be injected into the residual stream.

    Args:
        model (nn.Module): The model to which the hook will be attached.
        source (str): The source of the model ('timm' or 'transformers' usually).
        model_type (str): The type of the model ('APE' or 'RoPE').
        SAE (nn.Module): The SAE model used for feature reconstruction.
        features_to_remove (list): List of feature indices to be removed.
        
        top_features (bool): Whether to remove the top features based on the selectivity scores established by previous experiments. 
        top_features (bool) continued: If False, the features specified in features_to_remove will be removed. Default is True.
        top_features (bool) final: Previous selectivity experiments MUST be run for this to work, and the results must be saved to the JSON file.
        
        random_features (bool): Whether to randomly select features to remove. Default is False.
        k (int): Number of features to randomly select if random_features is True or top_features is True. Default is 0. Setting k to 0 is equivalent to just running the model with SAE reconstruction.

    Returns:
        RemovableHandle: A handle that can be used to remove the hook later.
    """

    assert not (top_features and random_features), "Cannot set both top_features and random_features to True."
    assert not (top_features and features_to_remove is not None), "Cannot set both top_features and features_to_remove."
    assert not (random_features and features_to_remove is not None), "Cannot set both random_features and features_to_remove."
    assert not (top_features and selectivity_scores_path is None), "selectivity_scores_path must be provided when top_features is True."
    assert not (not top_features and selectivity_scores_path is not None), "selectivity_scores_path should only be provided when top_features is True. Maybe you forgot to set top_features to True?"
    assert not ((random_features or top_features) and k < 0), "k must be greater than or equal to 0 when random_features or top_features is True."

    if random_features:
        # Randomly select features to remove
        features_to_remove = torch.randperm(SAE.W_enc.shape[-1])[:k].tolist()

    if top_features:
        # Load the selectivity scores from the JSON file
        import json
        with open(selectivity_scores_path, "r") as f:
            selectivity_scores = json.load(f)
        key = f"{model_type}_layer_{layer}"
        TSFP = selectivity_scores[key] # Top Selective Features by Position

        top_unique_positional_features_list = []
        for i in TSFP.values():
            for j in i:
                if j not in top_unique_positional_features_list:
                 top_unique_positional_features_list.append(j)

        features_to_remove = top_unique_positional_features_list[:k]
        

    # Attach the feature ablation hook
    handle = attach_feature_ablation_hook(
        SAE=SAE,
        model=model,
        source=source,
        features_to_remove=features_to_remove,
        layer=layer,
        block=block
    )

    return handle