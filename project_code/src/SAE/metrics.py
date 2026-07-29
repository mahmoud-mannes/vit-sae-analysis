"""Evaluation metrics for sparse autoencoders.

These replace the single batch ``EV`` proxy in ``train_SAE.py`` with held out
numbers that actually discriminate a good SAE from a bad one:

* **FVU** (fraction of variance unexplained), the honest reconstruction metric.
  ``FVU = ||x - x_hat||^2 / ||x - mean(x)||^2``. Explained variance is ``1 - FVU``.
* **L0**, the mean number of active latents per token (the realised sparsity).
* **dead fraction**, latents that never fire over the evaluation set.
* **cosine**, mean cosine similarity between ``x`` and ``x_hat``.
* **ground truth recovery** (synthetic only): how well the learned dictionary
  matches the true features, via mean max cosine similarity.
* **downstream delta** (real ViT only): the change in the model's own output
  when the layer activation is replaced by its SAE reconstruction. This is the
  metric that tells you the SAE kept what the model uses, not just what looks
  big in L2.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def reconstruction_metrics(sae, batches) -> dict:
    """FVU, explained variance, L0, dead fraction and cosine over ``batches``."""
    was_training = sae.training
    sae.eval()

    d_hidden = sae.d_hidden
    ever_fired = torch.zeros(d_hidden, dtype=torch.bool)
    sse = 0.0          # sum of squared reconstruction error
    l0_sum = 0.0
    cos_sum = 0.0
    n = 0
    xs = []
    for x in batches:
        x_hat, z, _ = sae(x)
        sse += ((x - x_hat) ** 2).sum().item()
        l0_sum += (z > 0).float().sum(dim=-1).sum().item()
        cos_sum += F.cosine_similarity(x, x_hat, dim=-1).sum().item()
        ever_fired |= (z > 0).any(dim=0).cpu()
        n += x.shape[0]
        xs.append(x)

    x_all = torch.cat(xs, dim=0)
    denom = ((x_all - x_all.mean(dim=0)) ** 2).sum().item()
    fvu = sse / (denom + 1e-12)

    if was_training:
        sae.train()

    return {
        "fvu": fvu,
        "explained_variance": 1.0 - fvu,
        "l0": l0_sum / n,
        "dead_fraction": float((~ever_fired).float().mean()),
        "cosine": cos_sum / n,
        "n_tokens": n,
    }


@torch.no_grad()
def mean_max_cosine(learned_dirs: torch.Tensor, true_dirs: torch.Tensor) -> dict:
    """Mean max cosine similarity between two sets of directions.

    ``recall`` asks, for each true feature, how close is the best learned
    feature. ``precision`` asks the reverse. Both live in ``[0, 1]``; higher is
    better recovery of the ground truth dictionary.
    """
    a = F.normalize(learned_dirs, dim=1)
    b = F.normalize(true_dirs, dim=1)
    sim = a @ b.t()  # [n_learned, n_true]
    recall = sim.max(dim=0).values.mean().item()      # best learned per true
    precision = sim.max(dim=1).values.mean().item()   # best true per learned
    return {"recall_mmcs": recall, "precision_mmcs": precision}


@torch.no_grad()
def downstream_delta(model, source, block, sae, pixel_batches, scale: float = 1.0,
                     n_prefix: int = 0, device: str = "cpu") -> dict:
    """Change in the ViT output when a block's input activation is replaced by
    its SAE reconstruction (a reconstruction splice).

    Returns top-1 agreement with the clean run and the mean KL of the spliced
    logits from the clean logits. This is the faithfulness check: an SAE can have
    a low FVU yet still destroy the directions the model reads from, and only a
    downstream metric catches that.

    ``block`` is the ViT block module whose *input* is spliced. ``pixel_batches``
    is an iterable of preprocessed input tensors (transformers pixel_values, or
    timm image tensors). ``scale`` is the activation store's normalisation
    constant, so the SAE (trained on normalised activations) is fed and read in
    its own units. ``n_prefix`` prefix tokens (the class token) are left
    untouched, because the SAE is trained on patch tokens only; splicing the
    class token would feed the SAE a distribution it never saw and unfairly
    destroy the very token the classifier head reads.
    """
    def forward_logits(px):
        if source == "transformers":
            return model(pixel_values=px).logits
        return model(px)

    splice = {"on": False}

    def splice_hook(module, inputs):
        if not splice["on"]:
            return None
        x = inputs[0]
        patches = x[:, n_prefix:, :]
        shape = patches.shape
        flat = patches.reshape(-1, shape[-1]).to(torch.float32) * scale
        x_hat, _, _ = sae(flat)
        x_hat = (x_hat / scale).reshape(shape).to(x.dtype)
        spliced = torch.cat([x[:, :n_prefix, :], x_hat], dim=1)
        return (spliced,) + tuple(inputs[1:])

    handle = block.register_forward_pre_hook(splice_hook)
    agree, kl_sum, n = 0, 0.0, 0
    try:
        for px in pixel_batches:
            px = px.to(device)
            splice["on"] = False
            clean = forward_logits(px).float()
            splice["on"] = True
            spliced = forward_logits(px).float()
            agree += (clean.argmax(-1) == spliced.argmax(-1)).sum().item()
            kl_sum += F.kl_div(
                F.log_softmax(spliced, -1), F.softmax(clean, -1), reduction="sum"
            ).item()
            n += px.shape[0]
    finally:
        handle.remove()
    return {"top1_agreement": agree / max(1, n), "logit_kl": kl_sum / max(1, n), "n_images": n}
