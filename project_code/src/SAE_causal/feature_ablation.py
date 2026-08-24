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
    Attaches a feature ablation hook to the specified model.

    Args:
        model (nn.Module): The model to which the hook will be attached.
        source (str): The source of the model ('timm' or 'transformers' usually).
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
        features_to_remove = [randint(0, SAE.W_enc.shape[-1]) for _ in range(k)]

    if top_features:
        # Load the selectivity scores from the JSON file
        import json
        with open(selectivity_scores_path, "r") as f:
            selectivity_scores = json.load(f)

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