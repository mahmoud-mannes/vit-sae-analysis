"""Activation store for SAE training.

For more details, see the docstring of ``ActivationStore`` in ``activation_store.py``. The only
difference is that this class keeps the activations in memory and uses indices to access them, rather than copying them into separate tensors.
This is useful for very large runs where the activations are stored in a memory-mapped ``.npy`` file.
"""

from __future__ import annotations

import math
import torch

class ActivationStoreMemmap:
    """
    An activation store that keeps the activations in memory and uses indices to access them, rather than copying them into separate tensors.
    This is useful for very large runs where the activations are stored in a memory-mapped ``.npy`` file.

    In order to avoid large random access patterns, the mean of the training activations is computed from a sample of the training activations, rather than the entire training set.
    Additionally, the training and validation sets are defined by indices, rather than copying the activations into separate tensors.
    We define these indices in terms of the image index, rather than the token index, to avoid large random access patterns when the activations are stored in a memory-mapped ``.npy`` file.
    """
    def __init__(
        self,
        number_images,
        activations: torch.Tensor,
        val_fraction: float = 0.05,
        normalize: bool = True,
        seed: int = 0,
        num_mean_samples: int = 50000,
    ) -> None:
        if activations.dim() != 2:
            raise ValueError(f"expected [n_tokens, d_model], got {tuple(activations.shape)}")
        self.d_model = activations.shape[1]
        self.number_images = number_images
        self.tokens_per_image = activations.shape[0] // number_images

        assert self.tokens_per_image * number_images == activations.shape[0], "activations must be divisible by number_images"

        # scalar normalisation so E[||x||] = sqrt(d_model)
        self.scale = 1.0
        if normalize:
            mean_norm = activations.norm(dim=1).mean().clamp_min(1e-8)
            self.scale = math.sqrt(self.d_model) / float(mean_norm)

        self.activations = activations
        self.act = self.activations.reshape(self.number_images, self.tokens_per_image, self.d_model)
        
        n = self.act.shape[0]
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g)
        n_val = max(1, int(n * val_fraction))
        self._val_idx = perm[:n_val]
        self._train_idx = perm[n_val:]

        # mean of the normalised training activations from a sample, for b_dec init

        sample = self.act[self._train_idx[:num_mean_samples]].reshape(-1, self.d_model)
        self.mean = (sample * self.scale).mean(dim=0)
        self._seed = seed

    # ------------------------------------------------------------------ views
    @property
    def train(self) -> torch.Tensor:
        return self.act[self._train_idx].reshape(-1, self.d_model)

    @property
    def val(self) -> torch.Tensor:
        return self.act[self._val_idx].reshape(-1, self.d_model)

    def n_train(self) -> int:
        return self._train_idx.shape[0] * self.tokens_per_image

    # -------------------------------------------------------------- iteration
    def epochs(self, batch_size: int, n_epochs: int, device: str = "cpu"):
        """Yield shuffled training batches for ``n_epochs`` passes."""
        assert batch_size >= self.tokens_per_image, "batch_size must be at least tokens_per_image"
        n = self._train_idx.shape[0]
        images_per_batch = batch_size // self.tokens_per_image
        for epoch in range(n_epochs):
            g = torch.Generator().manual_seed(self._seed + epoch + 1)
            order = self._train_idx[
                torch.randperm(n, generator=g)
                ]
            for start in range(0, n, images_per_batch):
                idx = order[start : start + images_per_batch]
                batch = self.act[idx].reshape(-1, self.d_model).to(device)
                batch = batch * self.scale
                yield batch

    def num_steps(self, batch_size: int, n_epochs: int) -> int:
        images_per_batch = batch_size // self.tokens_per_image
        return math.ceil(self._train_idx.shape[0] / images_per_batch) * n_epochs

    def val_batches(self, batch_size: int, device: str = "cpu"):
        n = self._val_idx.shape[0]
        images_per_batch = batch_size // self.tokens_per_image
        for start in range(0, n, images_per_batch):
            yield self.act[self._val_idx[start : start + images_per_batch]].reshape(-1, self.d_model).to(device) * self.scale
