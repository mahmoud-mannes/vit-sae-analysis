""" This module visualizes the activations of features on a TxT grid to get a more intuitive
and visual feel for where these features activate most. We do this by:

a) extracting the top positional feature candidates using the top_candidates.py module
b) grouping latent SAE activations by position 
c) extracting the mean activation of the studied features across all positions
d) visualizing the mean activations on a TxT grid (excluding the CLS token)

"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
import common

def group_activations_by_position(activations, num_tokens_per_image=197):
  """ Takes a TxD tensor representing the activations of T image patches extracted from a model of
  dimension D, returns a dictionary with keys composed of integers from 0-196 representing
  the positions of the image patches. For each key x, the dictionary has a value containing
  all the image patches found in the position 'x'
  """
  mask_by_position = dict() # Dictionary of position masks for each position (where pos 0 is the CLS token)
  num_tokens = activations.shape[0]
  for i in range(num_tokens_per_image):
    mask = torch.arange(num_tokens, device = activations.device) % num_tokens_per_image == i 
    mask_by_position[i] = mask
  
  activations_by_position = {i:activations[mask_by_position[i]] for i in range(num_tokens_per_image)}
  return activations_by_position

def mean_feature_activation_by_position(grouped_activations, feature, num_tokens_per_image=197):
  """ Takes the grouped activations produced by the group_activations_by_position function,
  and the feature we desire to visualize.
  Returns the mean activations of that feature across all positions.
  
  """
  
  feature_list = list()
  for i in range(num_tokens_per_image):
    feature_list.append(grouped_activations[i].mean(dim=0)[feature].detach().cpu().item())
  return feature_list

def visualize_feature_activation(activations, feature, num_tokens_per_image=197, num_prefix_tokens=1):
  """ Utilizes the previous two functions to visualize feature activation on a TxT grid.
  """
  
  grouped_activations = group_activations_by_position(activations, num_tokens_per_image)
  feature_list = mean_feature_activation_by_position(grouped_activations, feature, num_tokens_per_image)
  
  grid_size = int((num_tokens_per_image - num_prefix_tokens) ** 0.5)
  feature_activations_by_position_matrix = np.array(feature_list[1:]).reshape((grid_size,grid_size)) # Exclude CLS token and reshape to a square matrix
  plt.matshow(feature_activations_by_position_matrix)
  path_to_save = os.path.abspath(os.path.join(common._FIGURES_dir, f"feature_{feature}_activation_by_position.png"))
  plt.savefig(path_to_save) 