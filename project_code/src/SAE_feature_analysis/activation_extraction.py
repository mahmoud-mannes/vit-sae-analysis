import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from experiments.common import load_imagenet
from main.prep_data import prep_data
from main.load_models import get_vit_blocks
from main.model import predict    

def activation_extraction(model, processor, source, layer, number_images, RPI = False, d_model=768, shuffle=False):
    """ Activation extraction from desired layer residual stream input

    layer: the layer from which input activations will be extracted 
    RPI: whether Random Permutation at Inference will be applied at inference

    by running inference both ways (with RPI and without RPI), we can better isolate
    the effect of the index of the image patch on the SAE feature activations.
    """

    # Define simple activation extraction hook
    activation_list = []
    def activation_extraction_hook(module, inputs, output):
        activation_list.append(inputs[0])

    # Extract model blocks and register hook
    blocks = get_vit_blocks(model, source)
    handle = blocks[layer].register_forward_hook(activation_extraction_hook)

    # Load imagenet and get dataloader
    dataset = load_imagenet()
    if shuffle:
        dataset = dataset.shuffle(buffer_size = number_images)
    DL = prep_data(dataset, processor, source, number_images = number_images, batch_size = 500, half = False)

    # Run inference
    predict(model, DL, source, half = False, RPI = RPI)
    handle.remove()

    return torch.cat(activation_list, dim=0).reshape(-1, d_model).contiguous()