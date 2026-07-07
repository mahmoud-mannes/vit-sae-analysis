""" Determine top feature candidates responsible for spatial structure in ViTs

This module separately analyses the mean activations of features on:
a) image patches that correspond to a specific position of our choice
b) the rest of the image patches

and calculates a score by subtracting the mean activations of b) from a), we then
rank the features by this score to determine the top feature candidates that may be
responsible for integrating spatial stucture in the models studied.

"""

import os
import sys
import torch


def get_top_candidates(latent_activations, target_position, num_tokens_per_image = 197, k=20):
    """ Analyse features and their mean activations to determine top candidate features
    responsible for spatial structure
    """
    # NOTE: pos 0 represents the CLS token, therefore it is not included in our analyses of positional features
    num_tokens = latent_activations.shape[0]

    mask = torch.arange(num_tokens, device = latent_activations.device) % num_tokens_per_image == target_position # this mask will allow us to select tokens at the target position

    mean_activation_target_position = latent_activations[mask].mean(0)
    mean_activation_elsewhere = latent_activations[~mask].mean(0)

    selectivity_score = mean_activation_target_position - mean_activation_elsewhere
    top_candidates = torch.topk(selectivity_score, k=k)

    return top_candidates