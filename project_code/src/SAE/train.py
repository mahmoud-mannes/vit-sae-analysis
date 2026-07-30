"""Trainer for the modern SAE.

One trainer drives every variant. The differences between variants live entirely
in the loss:

* ``relu_l1``   : reconstruction + L1 on the latents (the old recipe).
* ``topk``      : reconstruction + AuxK, sparsity comes from the top k rule.
* ``batchtopk`` : reconstruction + AuxK, sparsity comes from the batch top k rule.
* ``jumprelu``  : reconstruction + an L0 penalty through the JumpReLU STE, plus
                  AuxK to keep latents alive.

Dead latents are tracked by how many tokens have passed since each last fired.
AuxK (Gao et al., 2024) then asks the dead latents to reconstruct the residual,
which revives them without the hand tuned resampling in ``resample.py``.

The decoder is kept unit norm every step, and the component of the decoder
gradient parallel to the decoder directions is projected out first so the norm
constraint does not fight the optimiser.
"""

from __future__ import annotations

import math
import torch

from .sae import SAE, RectangleSTE, auxiliary_loss
from .metrics import reconstruction_metrics


def _lr_lambda(warmup_steps, total_steps):
    def f(step):
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return f


def train_sae(
    store,
    architecture: str = "batchtopk",
    d_hidden: int | None = None,
    expansion: int = 16,
    k: int | None = None,
    n_epochs: int = 20,
    batch_size: int = 4096,
    lr: float = 4e-4,
    l1_coef: float = 1e-3,
    l0_coef: float = 1e-3,
    aux_coef: float = 1.0 / 32.0,
    k_aux: int = 256,
    warmup_frac: float = 0.05,
    dead_tokens_threshold: int = 200_000,
    use_b_dec: bool = True,
    jumprelu_bandwidth: float = 0.3,
    jumprelu_init_threshold: float = 0.05,
    eval_every: int = 200,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = False,
):
    torch.manual_seed(seed)
    d_model = store.d_model
    d_hidden = d_hidden or d_model * expansion

    sae = SAE(
        d_model, d_hidden, architecture=architecture, k=k,
        mean=store.mean if use_b_dec else None, use_b_dec=use_b_dec,
        jumprelu_bandwidth=jumprelu_bandwidth,
        jumprelu_init_threshold=jumprelu_init_threshold,
    ).to(device)
    if not use_b_dec:
        sae.b_dec.requires_grad_(False)

    opt = torch.optim.AdamW(sae.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    total_steps = store.num_steps(batch_size, n_epochs)
    warmup_steps = int(warmup_frac * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(warmup_steps, total_steps))

    tokens_since_fired = torch.zeros(d_hidden, device=device)
    history = []
    step = 0

    for x in store.epochs(batch_size, n_epochs, device):
        sae.train()
        x_hat, z, pre = sae(x)
        recon = ((x_hat - x) ** 2).sum(dim=-1).mean()
        loss = recon

        if architecture == "relu_l1":
            loss = loss + l1_coef * z.abs().sum(dim=-1).mean()
        elif architecture in ("topk", "batchtopk"):
            dead_mask = tokens_since_fired > dead_tokens_threshold
            loss = loss + aux_coef * auxiliary_loss(sae, x, x_hat, pre, dead_mask, k_aux)
        elif architecture == "jumprelu":
            theta = torch.exp(sae.log_threshold)
            gate = RectangleSTE.apply(pre, theta, sae.jumprelu_bandwidth)
            loss = loss + l0_coef * gate.sum(dim=-1).mean()
            dead_mask = tokens_since_fired > dead_tokens_threshold
            loss = loss + aux_coef * auxiliary_loss(sae, x, x_hat, pre, dead_mask, k_aux)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        sae.remove_parallel_grad()
        opt.step()
        sched.step()
        sae.normalize_decoder()

        with torch.no_grad():
            fired = (z > 0).any(dim=0)
            tokens_since_fired += x.shape[0]
            tokens_since_fired[fired] = 0

        step += 1
        if verbose and step % eval_every == 0:
            m = reconstruction_metrics(sae, store.val_batches(batch_size, device))
            m["step"] = step
            m["loss"] = float(loss.detach().cpu())
            history.append(m)
            print(
                f"  step {step:5d}/{total_steps}  fvu {m['fvu']:.4f}  "
                f"L0 {m['l0']:.1f}  dead {m['dead_fraction']:.3f}",
                flush=True,
            )

    # BatchTopK deploys as a per token JumpReLU with a single global threshold
    # (the recommended conversion). Plain TopK stays as a per token top k rule,
    # so it is not thresholded here.
    if architecture == "batchtopk":
        sae.estimate_threshold(list(store.val_batches(batch_size, device)))

    final = history[-1] if history else {}
    return sae, {"history": history, "final": final}
