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

  handle = blocks[layer].register_forward_pre_hook(intervention_hook)

  return handle