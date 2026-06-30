import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import os
import sys

path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(path)

from resample import resample_dead_features, get_high_loss_examples

class SAE_Module(nn.Module):
  def __init__(self, d_model, d_multiplier):
    super().__init__()
    self.d_model = d_model
    self.d_multiplier = d_multiplier

    self.linear_up = nn.Sequential(
        nn.Linear(d_model, d_model * d_multiplier),
        nn.ReLU()
    )
    self.linear_down = nn.Linear(d_model * d_multiplier, d_model)
  def forward(self, x):
    latents = self.linear_up(x)
    return self.linear_down(latents), latents

class SAE_Data(Dataset):
  def __init__(self, data):
    super().__init__()
    self.data = data
  def __getitem__(self,idx):
    return self.data[idx]
  def __len__(self):
    return len(self.data)
  


def train_SAE(data, d_model, d_multiplier, sparsity_lambda, lr,  window_size, resample_every, batch_size, warmup_steps):

    d_hidden = d_model * d_multiplier

    data = SAE_Data(data)
    

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def LambdaLR(step):
       if step <= warmup_steps:
          return step / warmup_steps
       else:
          return 1

    SAE = SAE_Module(d_model, d_multiplier).to(device)
    optimizer = torch.optim.Adam(SAE.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer = optimizer, lr_lambda = LambdaLR)


    SAE_DL = DataLoader(data, batch_size = batch_size, pin_memory = True, num_workers = min(16, os.cpu_count()))

    ever_fired = torch.zeros(d_hidden, dtype=torch.bool)
    eps = 1e-6

    for step, x in enumerate(SAE_DL):
        x = x.to(device=device, dtype=torch.float32)

        x_reconstructed, latents = SAE(x)

        reconstruction_loss = nn.MSELoss()(x_reconstructed, x)
        sparsity_loss = sparsity_lambda * latents.abs().sum(dim=-1).mean()

        loss = sparsity_loss + reconstruction_loss
        print(loss.detach().cpu().item())

        optimizer.zero_grad()  
        loss.backward()
        optimizer.step()
        scheduler.step()

        fired_this_batch = (latents.detach() > eps).any(dim = 0).cpu()
        ever_fired = ever_fired | fired_this_batch

        if step % window_size == 0 and step > 0:
            dead_fraction = (~ever_fired).float().mean()
            print(f"step {step} dead fraction {dead_fraction:.3f}")
            l0 = (latents.detach() > 0).float().sum(dim=-1).mean()
            print(f"L0: {l0.item():.1f}")

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
                    print(f"  resampled {len(dead_indices)} dead features")


            ever_fired = torch.zeros(d_hidden, dtype=torch.bool)
