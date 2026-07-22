import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import datasets
import math
import numpy as np
from SAE_feature_analysis.activation_extraction import activation_extraction


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