"""Modern sparse autoencoder for ViT residual streams.

This module supersedes the vanilla ReLU + L1 autoencoder in ``train_SAE.py``.
It keeps a single ``SAE`` class whose sparsity mechanism is selected by the
``architecture`` argument, so the same training and evaluation code drives every
variant and the comparison is apples to apples.

Design choices, and why they matter for a ViT position study:

* **Pre-encoder bias (``b_dec``).** The encoder sees ``x - b_dec`` and the
  decoder adds ``b_dec`` back. ``b_dec`` is initialised to the mean activation.
  Residual streams have a large shared mean component; centring it stops that
  component from leaking into every feature.
* **Tied initialisation.** ``W_enc`` starts as ``W_dec.T`` and decoder rows start
  unit norm, which is a well tested starting point (Anthropic, 2023).
* **Unit norm decoder.** Decoder directions are kept unit norm so ``L0`` and the
  feature magnitudes are comparable across features and across variants.
* **Sparsity mechanisms.** ``relu_l1`` is the old baseline. ``topk`` and
  ``batchtopk`` set ``L0`` directly and avoid the activation shrinkage that an L1
  penalty causes. ``jumprelu`` learns a per feature threshold. Both ``topk`` and
  ``batchtopk`` dominate L1 on the measured runs (see docs/SAE_RESULTS.md);
  ``topk`` was cleanest at low L0 and ``batchtopk`` best at a larger budget.
  ``batchtopk`` converts to a single global threshold at inference and runs as a
  per token JumpReLU (Bussmann et al., 2024, arXiv:2412.06410).

Convention: activations are row vectors, ``x`` has shape ``[batch, d_model]``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ARCHITECTURES = ("relu_l1", "topk", "batchtopk", "jumprelu")


class RectangleSTE(torch.autograd.Function):
    """Straight through estimator for the JumpReLU gate and its L0 count.

    Forward is the Heaviside step ``H(pre - theta)``. The gradient with respect
    to ``theta`` uses a rectangular kernel of width ``bandwidth`` as a smooth
    surrogate for the Dirac delta, following Rajamanoharan et al. (2024,
    arXiv:2407.14435). The gradient with respect to ``pre`` is passed straight
    through as the gate value.
    """

    @staticmethod
    def forward(ctx, pre, theta, bandwidth):
        ctx.save_for_backward(pre, theta)
        ctx.bandwidth = bandwidth
        return (pre > theta).to(pre.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        pre, theta = ctx.saved_tensors
        bw = ctx.bandwidth
        # rectangular window centred on the threshold
        in_window = ((pre - theta).abs() <= (bw / 2.0)).to(pre.dtype)
        grad_theta = grad_output * (-1.0 / bw) * in_window
        # sum the theta gradient over the batch (theta is shared across tokens)
        grad_theta = grad_theta.sum(dim=0)
        return None, grad_theta, None


class SAE(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        architecture: str = "batchtopk",
        k: int | None = None,
        mean: torch.Tensor | None = None,
        use_b_dec: bool = True,
        jumprelu_bandwidth: float = 0.3,
        jumprelu_init_threshold: float = 0.05,
    ) -> None:
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}, got {architecture}")
        if architecture in ("topk", "batchtopk") and k is None:
            raise ValueError(f"architecture {architecture!r} needs an integer k (target L0)")

        self.d_model = d_model
        self.d_hidden = d_hidden
        self.architecture = architecture
        self.k = k
        self.jumprelu_bandwidth = jumprelu_bandwidth

        # decoder rows are the feature directions, initialised to unit norm
        W_dec = torch.randn(d_hidden, d_model)
        W_dec = W_dec / W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(W_dec.t().clone())  # tied init
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))

        b_dec = torch.zeros(d_model) if mean is None else mean.detach().clone().float()
        self.b_dec = nn.Parameter(b_dec)

        # learnable log threshold for jumprelu; buffer threshold for batchtopk
        # inference. Kept for every variant so state dicts are uniform.
        self.log_threshold = nn.Parameter(
            torch.full((d_hidden,), float(torch.log(torch.tensor(jumprelu_init_threshold))))
        )
        self.register_buffer("inference_threshold", torch.zeros(d_hidden))
        # whether to use the fixed inference threshold (set True after
        # estimate_threshold for batchtopk) instead of the training time rule.
        self.use_inference_threshold = False

    # ------------------------------------------------------------------ core
    def preactivation(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def _apply_sparsity(self, pre: torch.Tensor) -> torch.Tensor:
        arch = self.architecture

        if arch == "relu_l1":
            return F.relu(pre)

        if arch == "topk":
            acts = F.relu(pre)
            if self.use_inference_threshold:
                # deploy as a JumpReLU with the estimated global threshold
                return acts * (acts > self.inference_threshold)
            return _topk_per_token(acts, self.k)

        if arch == "batchtopk":
            acts = F.relu(pre)
            if self.use_inference_threshold:
                # deploy as a JumpReLU with the estimated global threshold
                return acts * (acts > self.inference_threshold)
            return _batch_topk(acts, self.k)

        if arch == "jumprelu":
            acts = F.relu(pre)
            theta = torch.exp(self.log_threshold)
            gate = RectangleSTE.apply(pre, theta, self.jumprelu_bandwidth)
            return acts * gate

        raise AssertionError(arch)

    def forward(self, x: torch.Tensor):
        """Returns ``(x_hat, z, pre)``.

        ``z`` is the sparse latent code, ``pre`` is the pre-sparsity activation
        (used by the auxiliary loss to revive dead features).
        """
        pre = self.preactivation(x)
        z = self._apply_sparsity(pre)
        x_hat = self.decode(z)
        return x_hat, z, pre

    # ------------------------------------------------------ decoder constraint
    @torch.no_grad()
    def normalize_decoder(self) -> None:
        norm = self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.W_dec.div_(norm)

    @torch.no_grad()
    def remove_parallel_grad(self) -> None:
        """Project the component of the decoder gradient that is parallel to the
        decoder directions out, so the unit norm constraint does not fight the
        optimiser. No op if there is no gradient yet."""
        if self.W_dec.grad is None:
            return
        w = self.W_dec
        g = self.W_dec.grad
        parallel = (g * w).sum(dim=1, keepdim=True) * w
        g.sub_(parallel)

    # ------------------------------------------------------ threshold estimate
    @torch.no_grad()
    def estimate_threshold(self, batches) -> None:
        """Estimate a global JumpReLU threshold from a batchtopk / topk model.

        For each batch we record the smallest activation that batchtopk kept.
        The inference threshold is the mean of those per batch minima, following
        the conversion recommended in the BatchTopK paper. After this the model
        runs as a per token JumpReLU with a fixed threshold.
        """
        mins = []
        was_training = self.training
        self.eval()
        self.use_inference_threshold = False
        for x in batches:
            pre = self.preactivation(x)
            acts = F.relu(pre)
            if self.architecture == "batchtopk":
                kept = _batch_topk(acts, self.k)
            else:
                kept = _topk_per_token(acts, self.k)
            positive = kept[kept > 0]
            if positive.numel() > 0:
                mins.append(positive.min())
        if mins:
            thr = torch.stack(mins).mean()
            self.inference_threshold.fill_(float(thr))
        self.use_inference_threshold = True
        if was_training:
            self.train()


# --------------------------------------------------------------------- helpers
def _topk_per_token(acts: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the top k activations per row, zero the rest."""
    k = min(k, acts.shape[-1])
    if k <= 0:
        return torch.zeros_like(acts)
    vals, idx = torch.topk(acts, k, dim=-1)
    out = torch.zeros_like(acts)
    out.scatter_(-1, idx, vals)
    return out


def _batch_topk(acts: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the top ``k * batch`` activations across the whole batch, zero rest.

    This lets some tokens use more than k latents and others fewer, while the
    average L0 stays at k. Ties at the boundary keep every equal value, so the
    realised L0 can exceed k by a hair; that matches the reference behaviour.
    """
    b = acts.shape[0]
    total = min(k * b, acts.numel())
    if total <= 0:
        return torch.zeros_like(acts)
    flat = acts.reshape(-1)
    vals, idx = torch.topk(flat, total)
    out = torch.zeros_like(flat)
    out.scatter_(0, idx, vals)
    return out.reshape_as(acts)


def auxiliary_loss(
    sae: SAE,
    x: torch.Tensor,
    x_hat: torch.Tensor,
    pre: torch.Tensor,
    dead_mask: torch.Tensor,
    k_aux: int = 512,
) -> torch.Tensor:
    """AuxK loss: reconstruct the main model's residual using only dead latents.

    This is the OpenAI TopK trick (Gao et al., 2024, arXiv:2406.04093) and it
    replaces dead feature resampling. Dead features are pushed to explain what
    the live features missed, which revives them without hand tuned resets.
    """
    if dead_mask.sum() == 0:
        return x.new_zeros(())
    residual = (x - x_hat).detach()
    # only dead features are eligible to fire in the auxiliary path
    dead_pre = pre.masked_fill(~dead_mask.unsqueeze(0), float("-inf"))
    dead_acts = F.relu(dead_pre)
    k_eff = min(k_aux, int(dead_mask.sum().item()))
    z_aux = _topk_per_token(dead_acts, k_eff)
    residual_hat = z_aux @ sae.W_dec  # no b_dec: residual is already centred
    return F.mse_loss(residual_hat, residual)
