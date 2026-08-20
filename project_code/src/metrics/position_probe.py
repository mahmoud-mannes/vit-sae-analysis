import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import datasets
import math
import numpy as np
from SAE_feature_analysis.activation_extraction import activation_extraction


DEFAULT_COORDINATE_PROBE_ALPHA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
DEFAULT_COORDINATE_PROBE_TEST_FRACTION = 0.2
DEFAULT_COORDINATE_PROBE_VAL_FRACTION = 0.25


class LinearProbe(nn.Module):
  """
  Basic Probe with one Linear Layer, takes in activations and predicts the position of each token from the activations.
  """
  def __init__(self, D=768, num_positions=197):
    super().__init__()
    self.Linear = nn.Linear(D,num_positions)
  def forward(self, x):
    return self.Linear(x)

class NonLinearProbe(nn.Module):
  """
  Non-Linear Probe with one hidden layer, takes in activations and predicts the position of each token from the activations.
  """
  def __init__(self, D=768, num_positions=197):
    super().__init__()
    self.Linear1 = nn.Linear(D,D)
    self.Linear2 = nn.Linear(D,num_positions)
  def forward(self, x):
    x = F.relu(self.Linear1(x))
    return self.Linear2(x)

class Data(Dataset):
  def __init__(self, data):
    super().__init__()
    self.data = data
  def __getitem__(self,item):
    return self.data[item]
  def __len__(self):
    return len(self.data)

def train_probe_chunk(
    probe: nn.Module,
    optimizer: torch.optim.Optimizer,
    history: dict,
    activations: torch.Tensor,
    batch_size: int = 1024,
    train_val_split: tuple = (0.8,0.2),
    device: str = "cuda"
    ) -> tuple[nn.Module, dict]:
  """
  Trains a positional probe on a particular chunk, returns the partially trained probe and history of validation accuracy.

  Expected shape for activations: (B,T,D) with B the number of images the activations are extracted from,
  T the number of tokens per image, D the dimension of the model.
  """

  B, T, C = activations.shape

  # Define data class and dataloader

  #num_workers = min(os.cpu_count(), 16)
  num_workers = 0

  data = Data(activations)

  data_tr, data_val = random_split(data, train_val_split)

  DL_tr = DataLoader(data_tr, batch_size = batch_size, shuffle = True, num_workers = num_workers)
  DL_val = DataLoader(data_val, batch_size = batch_size, shuffle = False, num_workers = num_workers)


  for batch in DL_tr:
    # Prevent gradient accumulation

    optimizer.zero_grad()

    # Get logits, evaluate loss

    batch = batch.float().to(device)
    logits = probe(batch)
    logits = logits.reshape(-1, T)
    targets = torch.arange(T, device=device)
    targets = targets.expand(batch.shape[0], -1)
    targets = targets.reshape(-1)
    loss = F.cross_entropy(logits,targets)

    # Backward pass, optimizer step

    loss.backward()
    optimizer.step()

  loss_running = 0
  acc_running = 0
  num_batches = math.ceil(len(data_val) / batch_size)
  # Evaluating the model on the validation set
  with torch.inference_mode():
    for batch in DL_val:
      batch = batch.float().to(device)

      logits = probe(batch)
      targets = torch.arange(T, device=device)
      targets = targets.expand(batch.shape[0], -1)

      preds = logits.argmax(dim=-1)
      correct = (preds == targets).sum()
      acc_running += (correct / (batch.shape[1] * batch.shape[0])) / num_batches

      logits = logits.reshape(-1, T)
      targets = targets.reshape(-1)
      loss = F.cross_entropy(logits, targets)

      loss_running += loss / num_batches
  history["loss"].append(loss_running.detach().cpu().item())
  history["accuracy"].append(acc_running.detach().cpu().item())

  return probe, history

def train_probe_streaming(
    model,
    processor,
    source,
    dataset: datasets.DatasetDict,
    layer: int,
    probe_type: str = "linear",
    num_passes:int = 10,
    lr: float = 1e-3,
    batch_size: int = 1024,
    weight_decay: float = 1e-4,
    num_images_per_chunk: int = 1000,
    threshhold_number_images: int = 10000,
    device: str = "cuda"
) -> tuple[nn.Module, dict]:
  """
  Utilizes the train_probe_chunk function to fully train the probe from start to finish, this is done by loading chunks of activations extracted
  from the ViT and feeding them into the train_probe_chunk. This is done to avoid the massive memory demands of storing tens of thousands of activations,
  and the storage demands of storing those activations to disk.
  This function can be particularly slow, due to the fact that inference with a ViT is required at every pass.

  Another important note is that this function does not fully separate the training and validation sets, it simply streams through the dataset and trains on all of the activations.
  For a fully accurate reproduction of our results, we recommend using the train_probe_memmap function, which uses a memory map file to store the activations and then trains on them in a more traditional manner.
  
  probe_type: str, either "linear" or "nonlinear", determines the type of probe to be used. Linear probes are the default probes used in most of our experiments.
  Non-linear probes are used to gauge whether positional information is available in the activations, but not linearly separable as is often the case in RoPE models. 
  """
  assert probe_type in ["linear","nonlinear"], f"probe_type must be either 'linear' or 'nonlinear', got {probe_type!r}"
  assert num_images_per_chunk <= threshhold_number_images, f"num_images_per_chunk must be less than or equal to threshhold_number_images, got {num_images_per_chunk} and {threshhold_number_images}"
  history = {
      "loss": [],
      "accuracy": []
  }


  num_images_test = 5
  acts = activation_extraction(
      model,
      processor,
      source,
      layer=layer,
      number_images=num_images_test , # Extracting 5 images just to get the dimension of the model
      RPI=False,
      dataset=dataset)

  acts = acts.view(num_images_test,-1,acts.shape[-1]).contiguous()
  if probe_type == "linear":
    probe = LinearProbe(acts.shape[-1], acts.shape[1]).to(device)
  else:
    probe = NonLinearProbe(acts.shape[-1], acts.shape[1]).to(device)
  optimizer = torch.optim.AdamW(
    probe.parameters(),
    lr=lr,
    weight_decay=weight_decay)

  num_chunks = math.floor(threshhold_number_images / num_images_per_chunk)

  for i in range(num_passes):
    for j in range(num_chunks):
        acts = activation_extraction(
          model,
          processor,
          source,
          layer=layer,
          number_images=num_images_per_chunk,
          RPI=False,
          shuffle=True,
          dataset=dataset)

        acts = acts.view(num_images_per_chunk,-1,acts.shape[-1])

        probe, history = train_probe_chunk(probe=probe, history=history,optimizer=optimizer,batch_size=batch_size,activations=acts.cpu())

  return probe, history



def train_probe_memmap(
    acts: np.memmap,
    probe_type: str = "linear",
    num_passes:int = 10,
    lr: float = 1e-3,
    batch_size: int = 1024,
    weight_decay: float = 1e-4,
    device: str = "cuda"
) -> tuple[nn.Module, dict]:
  """
  Utilizes the train_probe_chunk function to fully train the probe from start to finish. This is done by loading chunks of activations from a memory
  map file (in our case, created with numpy). This way, we avoid loading all of the activations into memory at once, and avoid the slowness that comes with streaming
  the activations from a ViT running inference at each chunk.

  probe_type: str, either "linear" or "nonlinear", determines the type of probe to be used. Linear probes are the default probes used in most of our experiments.
  Non-linear probes are used to gauge whether positional information is available in the activations, but not linearly separable as is often the case in RoPE models. 
  """
  assert probe_type in ["linear","nonlinear"], f"probe_type must be either 'linear' or 'nonlinear', got {probe_type!r}"
  
  history = {
      "loss": [],
      "accuracy": []
  }

  if probe_type == "linear":
    probe = LinearProbe(acts.shape[-1], acts.shape[1]).to(device)
  else:
    probe = NonLinearProbe(acts.shape[-1], acts.shape[1]).to(device)
  optimizer = torch.optim.AdamW(
    probe.parameters(),
    lr=lr,
    weight_decay=weight_decay)

  num_batches = math.ceil( acts.shape[0] / batch_size )
  for i in range(num_passes):
    index = 0
    for j in range(num_batches):
      try:
        activations = torch.from_numpy(acts[index: index + batch_size])
      except:
        activations = torch.from_numpy(acts[index:])
      probe, history = train_probe_chunk(probe=probe, history=history,optimizer=optimizer,batch_size=batch_size,activations=activations)
      index += batch_size

  return probe, history


def normalized_square_grid_coordinates(num_patches: int) -> tuple[np.ndarray, np.ndarray]:
  """
  Returns normalized row and column coordinates for a square patch grid.
  """
  grid_size = int(round(math.sqrt(num_patches)))
  if grid_size * grid_size != int(num_patches):
    raise ValueError(f"patch token count must be a square, got {num_patches}")
  if grid_size == 1:
    zeros = np.zeros(1, dtype=np.float32)
    return zeros, zeros
  denom = float(grid_size - 1)
  rows = np.repeat(np.arange(grid_size, dtype=np.float32) / denom, grid_size)
  cols = np.tile(np.arange(grid_size, dtype=np.float32) / denom, grid_size)
  return rows, cols


def split_image_indices(
    num_images: int,
    seed: int,
    test_fraction: float = DEFAULT_COORDINATE_PROBE_TEST_FRACTION,
    val_fraction: float = DEFAULT_COORDINATE_PROBE_VAL_FRACTION,
) -> dict[str, np.ndarray]:
  """
  Deterministically splits images into train/validation/test sets.
  """
  if num_images < 4:
    raise ValueError("need at least 4 images for train/validation/test splits")

  rng = np.random.default_rng(int(seed))
  order = rng.permutation(int(num_images))

  test_count = max(1, int(round(num_images * test_fraction)))
  if test_count >= num_images:
    test_count = num_images - 1

  remaining = num_images - test_count
  if remaining < 2:
    raise ValueError("not enough images left for train and validation after test split")

  val_count = max(1, int(round(remaining * val_fraction)))
  if val_count >= remaining:
    val_count = remaining - 1

  train_count = remaining - val_count
  if train_count < 1:
    raise ValueError("image split produced an empty train set")

  train = np.sort(order[:train_count])
  val = np.sort(order[train_count : train_count + val_count])
  test = np.sort(order[train_count + val_count :])
  return {"train": train, "val": val, "test": test}


def _flatten_image_tokens(layer_outputs: np.ndarray, image_indices: np.ndarray) -> np.ndarray:
  subset = layer_outputs[np.asarray(image_indices, dtype=int)]
  return subset.reshape(-1, subset.shape[-1]).astype(np.float64, copy=False)


def _repeat_targets(targets: np.ndarray, image_indices: np.ndarray) -> np.ndarray:
  return np.tile(targets.astype(np.float64, copy=False), len(image_indices))


def _normalized_alpha_grid(alpha_grid: tuple[float, ...]) -> list[float]:
  values = [float(alpha) for alpha in alpha_grid]
  if not values:
    raise ValueError("alpha_grid must not be empty")
  return values


def _prepare_ridge_problem(
    train_x: np.ndarray, eval_sets: dict[str, np.ndarray]
) -> dict[str, object]:
  x_mean = train_x.mean(axis=0, keepdims=True)
  x_std = train_x.std(axis=0, keepdims=True)
  x_std[x_std < 1e-6] = 1.0
  x_scaled = (train_x - x_mean) / x_std
  gram = np.dot(x_scaled.T, x_scaled)
  eigvals, eigvecs = np.linalg.eigh(gram)

  return {
      "train_scaled": x_scaled,
      "x_mean": x_mean[0],
      "x_std": x_std[0],
      "eigvals": eigvals,
      "eigvecs": eigvecs,
      "eval_bases": {
          name: np.dot((eval_x - x_mean) / x_std, eigvecs)
          for name, eval_x in eval_sets.items()
      },
  }


def _prepare_ridge_target(problem: dict[str, object], train_y: np.ndarray) -> dict[str, object]:
  y_mean = float(train_y.mean())
  y_centered = train_y - y_mean
  rhs = np.dot(problem["train_scaled"].T, y_centered)
  eig_rhs = np.dot(problem["eigvecs"].T, rhs)
  return {"y_mean": y_mean, "eig_rhs": eig_rhs}


def _predict_prepared_ridge(
    problem: dict[str, object], target_state: dict[str, object], alpha: float, eval_name: str
) -> np.ndarray:
  coeff = target_state["eig_rhs"] / (problem["eigvals"] + float(alpha))
  return np.dot(problem["eval_bases"][eval_name], coeff) + target_state["y_mean"]


def _select_best_alpha(
    alpha_grid: list[float],
    problem: dict[str, object],
    target_state: dict[str, object],
    val_y: np.ndarray,
    predict_fn,
) -> tuple[float, float]:
  best_alpha = alpha_grid[0]
  best_val = None
  for alpha in alpha_grid:
    val_pred = predict_fn(problem, target_state, alpha, "val")
    val_mse = float(np.mean((val_y - val_pred) ** 2))
    if best_val is None or val_mse < best_val:
      best_val = val_mse
      best_alpha = alpha
  return best_alpha, best_val


def _targets_for_split(
    targets: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  return (
      _repeat_targets(targets, train_idx),
      _repeat_targets(targets, val_idx),
      _repeat_targets(targets, test_idx),
  )


def _shuffled_targets(targets: np.ndarray, rng: np.random.Generator) -> np.ndarray:
  return targets[rng.permutation(targets.shape[0])]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
  """
  Computes held-out regression metrics for coordinate probe predictions.
  """
  y_true = np.asarray(y_true, dtype=np.float64)
  y_pred = np.asarray(y_pred, dtype=np.float64)
  mse = float(np.mean((y_true - y_pred) ** 2))
  denom = float(np.sum((y_true - y_true.mean()) ** 2))
  r2 = 1.0 - float(np.sum((y_true - y_pred) ** 2) / denom) if denom > 0 else 0.0
  if y_true.std() < 1e-12 or y_pred.std() < 1e-12:
    pearson = 0.0
  else:
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
  return {"mse": mse, "r2": r2, "pearson": pearson}


def fit_ridge_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    alpha_grid: tuple[float, ...] = DEFAULT_COORDINATE_PROBE_ALPHA_GRID,
) -> dict[str, float]:
  """
  Selects ridge alpha on validation data and reports held-out test metrics.
  """
  alpha_grid = _normalized_alpha_grid(alpha_grid)
  selection_problem = _prepare_ridge_problem(train_x, {"val": val_x})
  selection_target = _prepare_ridge_target(selection_problem, train_y)
  best_alpha, best_val = _select_best_alpha(
      alpha_grid,
      selection_problem,
      selection_target,
      val_y,
      _predict_prepared_ridge,
  )

  full_train_x = np.concatenate([train_x, val_x], axis=0)
  full_train_y = np.concatenate([train_y, val_y], axis=0)
  final_problem = _prepare_ridge_problem(full_train_x, {"test": test_x})
  final_target = _prepare_ridge_target(final_problem, full_train_y)
  test_pred = _predict_prepared_ridge(final_problem, final_target, best_alpha, "test")
  metrics = regression_metrics(test_y, test_pred)
  metrics["alpha"] = best_alpha
  metrics["validation_mse"] = best_val
  return metrics


def fit_shared_ridge_probes(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    targets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    alpha_grid: tuple[float, ...] = DEFAULT_COORDINATE_PROBE_ALPHA_GRID,
) -> dict[str, dict[str, float]]:
  """
  Fits several ridge targets while sharing one feature decomposition.
  """
  if torch.cuda.is_available() and train_x.shape[1] >= 128:
    return fit_shared_ridge_probes_torch(
        train_x, val_x, test_x, targets, alpha_grid=alpha_grid
    )

  alpha_grid = _normalized_alpha_grid(alpha_grid)
  selection_problem = _prepare_ridge_problem(train_x, {"val": val_x})
  selected = {}
  for name, (train_y, val_y, _test_y) in targets.items():
    target_state = _prepare_ridge_target(selection_problem, train_y)
    best_alpha, best_val = _select_best_alpha(
        alpha_grid,
        selection_problem,
        target_state,
        val_y,
        _predict_prepared_ridge,
    )
    selected[name] = {"alpha": best_alpha, "validation_mse": best_val}

  full_train_x = np.concatenate([train_x, val_x], axis=0)
  final_problem = _prepare_ridge_problem(full_train_x, {"test": test_x})
  results = {}
  for name, (train_y, val_y, test_y) in targets.items():
    full_train_y = np.concatenate([train_y, val_y], axis=0)
    target_state = _prepare_ridge_target(final_problem, full_train_y)
    choice = selected[name]
    test_pred = _predict_prepared_ridge(final_problem, target_state, choice["alpha"], "test")
    metrics = regression_metrics(test_y, test_pred)
    metrics.update(choice)
    results[name] = metrics
  return results


def _prepare_torch_ridge_problem(
    train_x: np.ndarray, eval_sets: dict[str, np.ndarray], device: torch.device
) -> dict[str, object]:
  train = torch.as_tensor(train_x, dtype=torch.float32, device=device)
  x_mean = train.mean(dim=0)
  x_std = train.std(dim=0, unbiased=False)
  x_std = torch.where(x_std < 1e-8, torch.ones_like(x_std), x_std)
  train_scaled = (train - x_mean) / x_std
  gram = train_scaled.T @ train_scaled
  eigvals, eigvecs = torch.linalg.eigh(gram)
  eigvals = eigvals.clamp_min_(0)
  return {
      "train_scaled": train_scaled,
      "eigvals": eigvals,
      "eigvecs": eigvecs,
      "eval_bases": {
          name: (
              (torch.as_tensor(value, dtype=torch.float32, device=device) - x_mean) / x_std
          ) @ eigvecs
          for name, value in eval_sets.items()
      },
  }


def _prepare_torch_ridge_target(
    problem: dict[str, object], train_y: np.ndarray, device: torch.device
) -> dict[str, object]:
  target = torch.as_tensor(train_y, dtype=torch.float32, device=device)
  y_mean = target.mean()
  rhs = problem["train_scaled"].T @ (target - y_mean)
  return {"y_mean": y_mean, "eig_rhs": problem["eigvecs"].T @ rhs}


def _predict_torch_ridge(
    problem: dict[str, object], target_state: dict[str, object], alpha: float, eval_name: str
) -> np.ndarray:
  coeff = target_state["eig_rhs"] / (problem["eigvals"] + float(alpha))
  prediction = problem["eval_bases"][eval_name] @ coeff + target_state["y_mean"]
  return prediction.detach().cpu().numpy().astype(np.float64, copy=False)


def fit_shared_ridge_probes_torch(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    targets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    alpha_grid: tuple[float, ...] = DEFAULT_COORDINATE_PROBE_ALPHA_GRID,
    device=None,
) -> dict[str, dict[str, float]]:
  """
  Torch implementation of shared-feature ridge probing for CPU or CUDA.
  """
  if device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
  device = torch.device(device)
  alpha_grid = _normalized_alpha_grid(alpha_grid)
  selection_problem = _prepare_torch_ridge_problem(train_x, {"val": val_x}, device)
  selected = {}
  for name, (train_y, val_y, _test_y) in targets.items():
    target_state = _prepare_torch_ridge_target(selection_problem, train_y, device)
    best_alpha, best_val = _select_best_alpha(
        alpha_grid,
        selection_problem,
        target_state,
        val_y,
        _predict_torch_ridge,
    )
    selected[name] = {"alpha": best_alpha, "validation_mse": best_val}

  del selection_problem
  if device.type == "cuda":
    torch.cuda.empty_cache()

  full_train_x = np.concatenate([train_x, val_x], axis=0)
  final_problem = _prepare_torch_ridge_problem(full_train_x, {"test": test_x}, device)
  results = {}
  for name, (train_y, val_y, test_y) in targets.items():
    full_train_y = np.concatenate([train_y, val_y], axis=0)
    target_state = _prepare_torch_ridge_target(final_problem, full_train_y, device)
    choice = selected[name]
    test_pred = _predict_torch_ridge(final_problem, target_state, choice["alpha"], "test")
    metrics = regression_metrics(test_y, test_pred)
    metrics.update(choice)
    results[name] = metrics

  del final_problem
  if device.type == "cuda":
    torch.cuda.empty_cache()
  return results


def evaluate_coordinate_probe(
    train_layer_outputs: np.ndarray,
    eval_layer_outputs: np.ndarray,
    split: dict[str, np.ndarray],
    shuffle_seed: int,
    alpha_grid: tuple[float, ...] = DEFAULT_COORDINATE_PROBE_ALPHA_GRID,
) -> dict[str, object]:
  """
  Evaluates held-out row and column probes on disjoint image and feature splits.
  """
  if train_layer_outputs.shape[1:] != eval_layer_outputs.shape[1:]:
    raise ValueError("train and eval layer outputs must share token and feature shapes")

  rows, cols = normalized_square_grid_coordinates(train_layer_outputs.shape[1])
  train_idx = split["train"]
  val_idx = split["val"]
  test_idx = split["test"]

  train_x = _flatten_image_tokens(train_layer_outputs, train_idx)
  val_x = _flatten_image_tokens(eval_layer_outputs, val_idx)
  test_x = _flatten_image_tokens(eval_layer_outputs, test_idx)

  train_row, val_row, test_row = _targets_for_split(rows, train_idx, val_idx, test_idx)
  train_col, val_col, test_col = _targets_for_split(cols, train_idx, val_idx, test_idx)

  rng = np.random.default_rng(int(shuffle_seed))
  shuffled_train_row = _shuffled_targets(train_row, rng)
  shuffled_val_row = _shuffled_targets(val_row, rng)
  shuffled_test_row = _shuffled_targets(test_row, rng)
  shuffled_train_col = _shuffled_targets(train_col, rng)
  shuffled_val_col = _shuffled_targets(val_col, rng)
  shuffled_test_col = _shuffled_targets(test_col, rng)

  probe_metrics = fit_shared_ridge_probes(
      train_x,
      val_x,
      test_x,
      {
          "row": (train_row, val_row, test_row),
          "column": (train_col, val_col, test_col),
          "row_null": (shuffled_train_row, shuffled_val_row, shuffled_test_row),
          "column_null": (shuffled_train_col, shuffled_val_col, shuffled_test_col),
      },
      alpha_grid=alpha_grid,
  )
  row_metrics = probe_metrics["row"]
  col_metrics = probe_metrics["column"]
  row_null = probe_metrics["row_null"]
  col_null = probe_metrics["column_null"]

  return {
      "n_train_images": int(len(train_idx)),
      "n_val_images": int(len(val_idx)),
      "n_test_images": int(len(test_idx)),
      "n_tokens_per_image": int(train_layer_outputs.shape[1]),
      "row": row_metrics,
      "column": col_metrics,
      "negative_control": {
          "row": row_null,
          "column": col_null,
      },
      "mean_test_r2": float((row_metrics["r2"] + col_metrics["r2"]) / 2.0),
      "negative_control_mean_test_r2": float((row_null["r2"] + col_null["r2"]) / 2.0),
  }
