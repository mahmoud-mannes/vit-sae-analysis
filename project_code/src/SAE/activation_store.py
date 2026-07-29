"""Activation store for SAE training.

The old pipeline concatenated every activation into one tensor and made a single
pass with no normalisation and no held out split. Three things fix that here:

* **Scalar normalisation.** All activations are scaled by one constant so that
  ``E[||x||] = sqrt(d_model)``. This is the standard SAE input normalisation. It
  makes the sparsity setting (L0, or the L1 coefficient) transfer across layers
  and models instead of depending on the raw activation scale. The mean is *not*
  subtracted here; the SAE learns it through ``b_dec``.
* **Train / validation split.** Metrics are reported on held out tokens so a low
  reconstruction error cannot be an artefact of memorising the training batch.
* **Shuffled multi epoch iteration.** Tokens are shuffled every epoch so batches
  mix positions and images.

The store is deliberately in memory and framework light. For very large runs
point ``from_file`` at a memory mapped ``.npy`` and the same interface holds.
"""

from __future__ import annotations

import math
import torch


class ActivationStore:
    def __init__(
        self,
        activations: torch.Tensor,
        val_fraction: float = 0.05,
        normalize: bool = True,
        seed: int = 0,
    ) -> None:
        if activations.dim() != 2:
            raise ValueError(f"expected [n_tokens, d_model], got {tuple(activations.shape)}")
        activations = activations.float()
        self.d_model = activations.shape[1]

        # scalar normalisation so E[||x||] = sqrt(d_model)
        self.scale = 1.0
        if normalize:
            mean_norm = activations.norm(dim=1).mean().clamp_min(1e-8)
            self.scale = math.sqrt(self.d_model) / float(mean_norm)
            activations = activations * self.scale

        n = activations.shape[0]
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g)
        n_val = max(1, int(n * val_fraction))
        self._val = activations[perm[:n_val]].contiguous()
        self._train = activations[perm[n_val:]].contiguous()

        # mean of the normalised training activations, for b_dec init
        self.mean = self._train.mean(dim=0)
        self._seed = seed

    # ------------------------------------------------------------------ views
    @property
    def train(self) -> torch.Tensor:
        return self._train

    @property
    def val(self) -> torch.Tensor:
        return self._val

    def n_train(self) -> int:
        return self._train.shape[0]

    # -------------------------------------------------------------- iteration
    def epochs(self, batch_size: int, n_epochs: int, device: str = "cpu"):
        """Yield shuffled training batches for ``n_epochs`` passes."""
        n = self._train.shape[0]
        for epoch in range(n_epochs):
            g = torch.Generator().manual_seed(self._seed + epoch + 1)
            order = torch.randperm(n, generator=g)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                yield self._train[idx].to(device)

    def num_steps(self, batch_size: int, n_epochs: int) -> int:
        return math.ceil(self._train.shape[0] / batch_size) * n_epochs

    def val_batches(self, batch_size: int, device: str = "cpu"):
        n = self._val.shape[0]
        for start in range(0, n, batch_size):
            yield self._val[start : start + batch_size].to(device)
