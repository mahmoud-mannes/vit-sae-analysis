import os
import sys
import torch
from torch.utils.hooks import RemovableHandle

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from main.load_models import get_vit_blocks, get_block_attention, get_block_mlp

def attach_feature_ablation_hook(
    SAE,
    model,
    source,
    features_to_remove: list,
    layer: int,
    block: str = None) -> RemovableHandle:

  """
  Attaches a feature ablation hook to the specified model. 

  Args:
      SAE (nn.Module): The SAE model used for feature reconstruction.
      model (nn.Module): The model to which the hook will be attached.
      source (str): The source of the model ('timm' or 'transformers' usually).
      features_to_remove (list): List of feature indices to be removed.
      layer (int): The layer of the model where the hook will be attached.
      block (str): The block of the model where the hook will be attached. Can be one of None, 'residual', 'attention', or 'mlp'. Default is None.
  
  Returns:
      RemovableHandle: A handle that can be used to remove the hook later.
  """
  assert block in [None, "residual", "mlp", "attention"], "block must be one of None, 'residual', 'attention', or 'mlp'"
  if block in ['residual', None]:
    def intervention_hook(module, input):
      if isinstance(input, tuple):
        residual = input[0].clone()
      else:
        residual = input.clone()

      latents = SAE.preactivation(residual)
      latents = SAE._apply_sparsity(latents)
      latents[:,:, features_to_remove] = 0.0

      residual_reconstructed = SAE.decode(latents)
      
      if isinstance(input, tuple):
        return (residual_reconstructed,) + input[1:]
  
      return residual_reconstructed
  else:
    def intervention_hook(module, input, output):
      if isinstance(output, tuple):
        block_output = output[0].clone()
      else:
        block_output = output.clone()

      latents = SAE.preactivation(block_output)
      latents = SAE._apply_sparsity(latents)
      latents[:,:, features_to_remove] = 0.0

      output_reconstructed = SAE.decode(latents)

      if isinstance(output,tuple):
        return (output_reconstructed,) + output[1:]
      
      return output_reconstructed

  if block in ['residual', None]:
    blocks = get_vit_blocks(model,source)
  elif block == 'attention':
    blocks = get_block_attention(model,source)
  elif block == 'mlp':
    blocks = get_block_mlp(model,source)

  if block in ['residual', None]:
    handle = blocks[layer].register_forward_pre_hook(intervention_hook)
  else:
    handle = blocks.register_forward_hook(intervention_hook)

  return handle