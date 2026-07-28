import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import os
import sys
import math
import time

path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(path)

from resample import resample_dead_features, get_high_loss_examples

class SAE_Module(nn.Module):
  """
  Sparse Autoencoder (SAE) module. This module consists of a linear layer that expands the input dimension to a higher-dimensional
  latent space, followed by a ReLU activation function, and then another linear layer that projects the latent representation back 
  to the original input dimension. The SAE is designed to learn sparse representations of the input data.
  """
  def __init__(self, d_model: int, d_multiplier: int) -> None:
    super().__init__()
    self.d_model = d_model
    self.d_multiplier = d_multiplier

    self.linear_up = nn.Sequential(
        nn.Linear(d_model, d_model * d_multiplier),
        nn.ReLU()
    )
    self.linear_down = nn.Linear(d_model * d_multiplier, d_model)
  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    latents = self.linear_up(x)
    return self.linear_down(latents), latents

class SAE_Data(Dataset):
  """
  A very basic dataset class for the Sparse Autoencoder (SAE) training. It takes a list of data samples and provides
  methods to access individual samples and the total number of samples in the dataset.
  """
  def __init__(self, data):
    super().__init__()
    self.data = data
  def __getitem__(self,idx):
    return self.data[idx]
  def __len__(self):
    return len(self.data)
  

def train_SAE(
    data, 
    d_model: int,
    d_multiplier: int,
    sparsity_lambda: float,
    lr: float,
    window_size: int,
    resample_every: int | float, 
    batch_size: int, 
    warmup_steps: int, 
    wandb_run,
    logging: bool = True,
    verbose: bool = False,
    metric_window_size: int = 10) -> SAE_Module:
    """
    Train the Sparse Autoencoder (SAE) on the given data. The training process involves optimizing the SAE to minimize
    a combination of reconstruction loss and sparsity loss. The function also tracks evaluation metrics over a specified 
    window size and periodically resamples dead features to improve the model's performance. This function
    also supports logging with wandb for monitoring training progress and evaluation metrics. This can be turned on or off
    by setting the logging parameter to True or False, respectively. 
    """
    d_hidden = d_model * d_multiplier

    # We define windows for tracking evaluation metrics. We will take the window size to be equal to 10 by default to allow for a more stable evaluation of the metrics, while still keeping recency of the values per metric log.
    EV_window = []
    L0_window = []

    data = SAE_Data(data)
    
    # Determine the device to use for training. If a GPU is available, it will be used; otherwise, the CPU will be used.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    total_steps = math.ceil(len(data) / batch_size)

    # Define lambda function for learning rate scheduling, warmup strategy is implemented for the first few steps of training. Then, cosine decay is applied to the learning rate for the remaining steps.

    def LambdaLR(step):
        if step < warmup_steps:
            return step / warmup_steps

        # Progress through the decay phase, normalized to [0, 1]
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        progress = min(progress, 1.0)

        # Cosine decay from 1.0 -> 0.0
        return 0.5 * (1 + math.cos(math.pi * progress))
      
    # Initialize the Sparse Autoencoder (SAE) model, optimizer, and learning rate scheduler.
    #  The SAE is moved to the appropriate device (GPU if available, otherwise CPU). The optimizer used is Adam,
    #  and the learning rate scheduler is defined using a lambda function that implements a warmup strategy for the 
    #  first few steps of training.
    SAE = SAE_Module(d_model, d_multiplier).to(device)
    optimizer = torch.optim.Adam(SAE.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer = optimizer, lr_lambda = LambdaLR)

    # Create a DataLoader for the training data. The DataLoader is responsible for batching the data and providing it to the model during training.
    SAE_DL = DataLoader(data, batch_size = batch_size, pin_memory = True, num_workers = 0)

    ever_fired = torch.zeros(d_hidden, dtype=torch.bool)
    eps = 1e-6

    start = time.time()
    for step, x in enumerate(SAE_DL):
        x = x.to(device=device, dtype=torch.float32)

        x_reconstructed, latents = SAE(x)

        reconstruction_loss = nn.MSELoss()(x_reconstructed, x)
        sparsity_loss = sparsity_lambda * latents.abs().sum(dim=-1).mean()

        loss = sparsity_loss + reconstruction_loss

        if verbose:
          print(loss.detach().cpu().item())

        optimizer.zero_grad()  
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
          W = SAE.linear_down.weight
          W /= W.norm(dim=0, keepdim=True).clamp_min(1e-8)

        # Track which features have fired (i.e., have non-zero activations) in the current batch.
        #  This is done by checking if any of the latent activations are greater than a small epsilon value.
        #  The `ever_fired` tensor keeps track of which features have fired across all batches seen so far, allowing us to
        #  identify "dead" features that have never activated.
        fired_this_batch = (latents.detach() > eps).any(dim = 0).cpu()
        ever_fired = ever_fired | fired_this_batch

        if step % window_size == 0 and step > 0:
            dead_fraction = (~ever_fired).float().mean()
            l0 = (latents.detach() > 0).float().sum(dim=-1).mean()
            if verbose:
              print(f"step {step} dead fraction {dead_fraction:.3f}")
              print(f"L0: {l0.item():.1f}")

            # Calculate the expected variance (EV) metric, which is a measure of how well the model is reconstructing 
            # the input data. The EV is calculated as 1 minus the ratio of the reconstruction loss to the mean squared 
            # value of the input data, with a small epsilon added to avoid division by zero. This metric is logged along
            # with the L0 metric, which measures the average number of non-zero elements in the latent representation, 
            # providing insight into the sparsity of the learned features.

            EV = 1 - (reconstruction_loss.detach().cpu().item() / (x.detach().cpu().var(unbiased=False).item() + eps))
            EV_window.append(EV)
            L0_window.append(l0.detach().cpu().item())

            if len(EV_window) > metric_window_size:
                EV_window.pop(0)
                L0_window.pop(0)
                if logging:
                    wandb_run.log({
                       "EV": sum(EV_window) / len(EV_window),
                        "L0": sum(L0_window) / len(L0_window), 
                        "dead_fraction": dead_fraction,
                       })

            # ---- resample dead features periodically ----

            if step % resample_every == 0:
                dead_indices = torch.nonzero(~ever_fired, as_tuple=True)[0].to(device)
                if len(dead_indices) > 0:
                    with torch.no_grad():
                        # re-run this batch's forward pass to get fresh reconstructions
                        # for picking high-loss examples (avoids relying on stale tensors)
                        x_recon_now, _ = SAE(x)
                    high_loss_examples = get_high_loss_examples(
                        x, x_recon_now, n_needed=len(dead_indices)
                    )
                    # if fewer high-loss examples than dead features, repeat with noise jitter
                    if high_loss_examples.shape[0] < len(dead_indices):
                        reps = (len(dead_indices) // high_loss_examples.shape[0]) + 1
                        high_loss_examples = high_loss_examples.repeat(reps, 1)[: len(dead_indices)]
                        high_loss_examples = high_loss_examples + 0.01 * torch.randn_like(high_loss_examples)

                    resample_dead_features(SAE, optimizer, dead_indices, high_loss_examples, device)
                    if verbose:
                        print(f"  resampled {len(dead_indices)} dead features")


            ever_fired = torch.zeros(d_hidden, dtype=torch.bool)
    end = time.time()
    wandb_run.log({
        "train_time": end - start
    })
    return SAE
