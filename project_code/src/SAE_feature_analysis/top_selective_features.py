import sys, os

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from SAE_feature_analysis.top_candidates import get_top_candidates
from SAE_feature_analysis.visualize_feature_activation import group_activations_by_position, mean_feature_activation_by_position
import numpy as np
import torch

eps=1e-6
def row_selectivity(matrix):
  """ Takes a TxT square matrix, returns a scalar indicating the row selectivity of the entries
  with values ranging from 1 to 10, with values greater than 4 indicating great selectivity.

  What this metric tries to capture is how much the mean activation of the most active row is
  greater than the mean activation of all rows. The higher this value, the more
  selective the feature is to a specific row in the grid.
  """
  matrix = torch.from_numpy(matrix).to(torch.float32)
  max_mean_row_activation = matrix.mean(dim = 1).max()
  matrix_mean = matrix.mean()

  score = (max_mean_row_activation) / (matrix_mean + eps)
  grid_size = len(matrix) - 1
  normalized_score = (score - 1) / (grid_size - 1) # A normalized version of the row selectivity score that ranges from 0 to 1, currently under testing.

  return score

def column_selectivity(matrix):
  """ Takes a TxT square matrix, returns a scalar indicating the column selectivity of the entries
  with values ranging from 1 to 10, with values greater than 4 indicating great selectivity.
  
  What this metric tries to capture is how much the mean activation of the most active column is
  greater than the mean activation of all columns. The higher this value, the more
  selective the feature is to a specific column in the grid.
  """
  return row_selectivity(matrix.transpose())

def top_selective_features(latent_activations, num_tokens=197, num_prefix_tokens=1, verbose=False):
    """
    Takes in a tensor of latent activations extracted from an SAE and returns the top selective features
    for each position in the grid. This is done in a loop over each position in the grid, where for each position, 
    the top candidates are extracted using their selectivity score (from the top_candidates.py module) and their selectivity 
    by row and column is calculated. The feature with the highest selectivity for each position is then returned.

    It is important to note that 'selectivity score' and 'row/column selectivity' are two different metrics. The selectivity score
    is a measure of how selective a feature is to a specific position in the grid, while the row/column selectivity is a 
    measure of how selective a feature is to a specific row or column in the grid.
    """
    
    TSFPD = dict() # Top Selective Features by Position Dictionary
    for i in range(num_tokens - 1):
        # Extract top candidates for the current position
        top = get_top_candidates(latent_activations, target_position=i + num_prefix_tokens, k = 3)

        grid_size = int(num_tokens ** 0.5)
        row = (i // grid_size)
        column = (i % grid_size)


        if verbose:
            print(f"{'-' *5} ROW {row} COLUMN {column} {'-' * 5}")

        FCSD = dict() # Feature Column Selectivity Dictionary
        FRSD = dict() # Feature Row Selectivity Dictionary
        for feature in top.indices:
            grouped_activations = group_activations_by_position(latent_activations)
            feature_list = mean_feature_activation_by_position(grouped_activations, feature = feature.item())
            MFAP = np.array(feature_list[num_prefix_tokens:]).reshape((grid_size,grid_size)) # Mean Feature Activation By Position matrix

            FCS,FRS = column_selectivity(MFAP), row_selectivity(MFAP) # Feature Column/Row Selectivity

            FCSD[FCS] = feature.item()
            FRSD[FRS] = feature.item()


        maximum_CS, maximum_RS = max(FCSD.keys()), max(FRSD.keys())
        candidate_column_feature = FCSD[maximum_CS]
        candidate_row_feature = FRSD[maximum_RS]

        TSFPD[(row,column)] = (candidate_row_feature, candidate_column_feature)

        if verbose:
            print(f"Feature {candidate_column_feature}, Column selectivity {maximum_CS}")
            print(f"Feature {candidate_row_feature}, Row selectivity {maximum_RS}")

    return TSFPD
        


    