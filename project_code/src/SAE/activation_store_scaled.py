"""Activation store for SAE training.

For more details, see the docstring of ``ActivationStore`` in ``activation_store.py``. The only
difference is that this class keeps the activations in memory and uses indices to access them, rather than copying them into separate tensors.
This is useful for very large runs where the activations are stored in a memory-mapped ``.npy`` file.
"""

from __future__ import annotations

import math
import torch

class ActivationStoreMemmap:
    def __init__(
        self,
        activations: torch.Tensor,
        val_fraction: float = 0.05,
        normalize: bool = True,
        seed: int = 0,
        num_mean_samples: int = 50000,
    ) -> None:
        if activations.dim() != 2:
            raise ValueError(f"expected [n_tokens, d_model], got {tuple(activations.shape)}")
        self.d_model = activations.shape[1]

        # scalar normalisation so E[||x||] = sqrt(d_model)
        self.scale = 1.0
        if normalize:
            mean_norm = activations.norm(dim=1).mean().clamp_min(1e-8)
            self.scale = math.sqrt(self.d_model) / float(mean_norm)

        self.activations = activations
        
        n = activations.shape[0]
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g)
        n_val = max(1, int(n * val_fraction))
        self._val_idx = perm[:n_val]
        self._train_idx = perm[n_val:]

        # mean of the normalised training activations from a sample, for b_dec init

        sample = activations[self._train_idx[:num_mean_samples]]
        self.mean = (sample * self.scale).mean(dim=0)
        self._seed = seed

    # ------------------------------------------------------------------ views
    @property
    def train(self) -> torch.Tensor:
        return self.activations[self._train_idx]

    @property
    def val(self) -> torch.Tensor:
        return self.activations[self._val_idx]

    def n_train(self) -> int:
        return self._train_idx.shape[0]

    # -------------------------------------------------------------- iteration
    def epochs(self, batch_size: int, n_epochs: int, device: str = "cpu"):
        """Yield shuffled training batches for ``n_epochs`` passes."""
        n = self._train_idx.shape[0]
        for epoch in range(n_epochs):
            g = torch.Generator().manual_seed(self._seed + epoch + 1)
            order = self._train_idx[
                torch.randperm(n, generator=g)
                ]
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                batch = self.activations[idx].to(device)
                batch = batch * self.scale
                yield batch

    def num_steps(self, batch_size: int, n_epochs: int) -> int:
        return math.ceil(self._train_idx.shape[0] / batch_size) * n_epochs

    def val_batches(self, batch_size: int, device: str = "cpu"):
        n = self._val_idx.shape[0]
        for start in range(0, n, batch_size):
            yield self.activations[self._val_idx[start : start + batch_size]].to(device)
