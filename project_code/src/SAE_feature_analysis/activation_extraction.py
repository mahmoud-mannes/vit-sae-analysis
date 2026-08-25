import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from experiments.common import load_imagenet
from main.prep_data import prep_data
from main.load_models import get_vit_blocks, get_block_attention, get_block_mlp
from main.model import predict    

def activation_extraction(
    model, 
    processor,
    source: str, 
    layer: int, 
    number_images: int, 
    RPI: bool = False, 
    d_model: int = 768, 
    shuffle: bool = False,  
    dataset=None) -> torch.Tensor:

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
    if not dataset:
        dataset = load_imagenet()
    if shuffle:
        dataset = dataset.shuffle(buffer_size = number_images)
    DL = prep_data(dataset, processor, source, number_images = number_images, batch_size = 500, half = False)

    # Run inference
    predict(model, DL, source, half = False, RPI = RPI)
    handle.remove()

    return torch.cat(activation_list, dim=0).reshape(-1, d_model).contiguous()

def activation_extraction_memmap(
    model,
    processor,
    source: str,
    layer: int,
    number_images: int,
    RPI: bool = False,
    d_model: int = 768,
    shuffle: bool = False,
    dataset=None, 
    block: str = None, 
    path: str = None, 
    verbose: bool = False) -> np.memmap:
    """ Activation extraction from desired block input, appending them to a memory-mapped file

    layer: the layer from which input activations will be extracted 
    RPI: whether Random Permutation at Inference will be applied at inference
    block: the block from which input activations will be extracted (residual, attention, mlp) with None defaulting
    to residual stream input

    by running inference both ways (with RPI and without RPI), we can better isolate
    the effect of the index of the image patch on the SAE feature activations.
    """

    assert block in [None, 'residual', 'attention', 'mlp'], "block must be one of None, 'residual', 'attention', or 'mlp'"

    if not path:
        path = f"residual_layer{layer}_inputs_test.bin"
    binary_file = open(path, "wb")

    # Define simple activation extraction hook for one of two cases, depending on whether we are extracting from the residual stream or from a specific block.
    # For the residual stream, we extract the input to the block, for attention and mlp blocks, we extract the output of the block.
    def register_activation(module, input, output):
        if block in [None, 'residual']:
            activation = input[0].detach().cpu().numpy().reshape(-1, input[0].shape[-1])
        else:
            if isinstance(output, tuple):
                activation = output[0].detach().cpu().numpy().reshape(-1, output[0].shape[-1])
            else:
                activation = output.detach().cpu().numpy().reshape(-1, output.shape[-1])
        binary_file.write(activation.tobytes())

    # Extract model blocks and register hook
    blocks = get_vit_blocks(model, source) # The block variable is used as an input to determine which block to extract activations from. In contrast, blocks is either the list of all blocks, or the specific block from which we are extracting activations, depending on the value of block.
    if block == 'attention':
        blocks = get_block_attention(blocks[layer], source)
    elif block == 'mlp':
        blocks, _ = get_block_mlp(blocks[layer], source)

    if not block or block == 'residual':
        handle = blocks[layer].register_forward_hook(register_activation)
    else:
        handle = blocks.register_forward_hook(register_activation)

    # Load imagenet and get dataloader
    if not dataset:
        dataset = load_imagenet()
    if shuffle:
        dataset = dataset.shuffle(buffer_size = number_images)
    
    DL = prep_data(dataset, processor, source, number_images = number_images, batch_size = 500, half = False)


    # Run inference
    predict(model, DL, source, half = False, RPI = RPI, verbose = verbose)
    handle.remove()
    binary_file.close()

    data = (np.memmap(path, dtype="float32", mode="r").reshape(-1, d_model))
    data_to_store = torch.from_numpy(data)
    if verbose:
        print(f"Data shape: {data_to_store.shape}")

    return data_to_store