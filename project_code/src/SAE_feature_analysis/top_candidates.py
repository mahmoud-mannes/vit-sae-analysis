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

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from experiments.common import load_imagenet
from SAE.train_SAE import SAE_Module
from main.prep_data import prep_data
from main.load_models import load_ape, load_rope, get_vit_blocks, get_block_mlp
from main.model import predict

def activation_extraction(model, processor, source, layer, number_images, RPI = False, d_model=768):
    """ Activation extraction from desired layer residual stream input

    layer: the layer from which input activations will be extracted 
    RPI: whether Random Permutation at Inference will be applied at inference

    by running inference both ways (with RPI and without RPI), we can better isolate
    the effect of the index of the image patch on the SAE feature activations.
    """

    device = model.device

    # Define simple activation extraction hook
    activation_list = []
    def activation_extraction_hook(module, inputs, output):
        activation_list.append(inputs[0])

    # Extract model blocks and register hook
    blocks = get_vit_blocks(model, source)
    handle = blocks[layer].register_forward_hook(activation_extraction_hook)

    # Load imagenet and get dataloader
    dataset = load_imagenet()
    DL = prep_data(dataset, processor, source, number_images = number_images, batch_size = 500, half = False)

    # Run inference
    predict(model, DL, source, half = False, RPI = RPI)
    handle.remove()

    return torch.cat(activation_list, dim=0).reshape(-1, d_model).contiguous()

    
    


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