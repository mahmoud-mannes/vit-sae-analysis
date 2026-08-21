"""Shared helpers for the experiment scripts: paths, dataset streaming, small
curve summaries, JSON IO, and plotting.

Everything here is deliberately thin so the scripts and the Colab notebook can
share the same building blocks.
"""

import json
import os
import sys

import numpy as np
import torch

# Make `main`, `metrics` and `interventions` importable no matter where a script
# is launched from.
SRC_ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

REPO_ROOT = os.path.abspath(SRC_ROOT + "/../..")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
REFERENCE_DIR = os.path.join(RESULTS_DIR, "reference")


def ensure_dirs():
    for d in (RESULTS_DIR, FIGURES_DIR, REFERENCE_DIR):
        os.makedirs(d, exist_ok=True)


def get_hf_token(token=None):
    return (
        token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def load_imagenet(
    split="validation",
    streaming=True,
    token=None,
    shuffle=False,
    seed=0,
    buffer_size=2000,
    dataset_id="ILSVRC/imagenet-1k",
    fallback_id="benjamin-paine/imagenet-1k-256x256",
):
    """Stream ImageNet-1k from the Hub, yielding {'image': PIL, 'label': int}.

    The official split (`ILSVRC/imagenet-1k`) is gated. If your account has not
    accepted its terms, the load returns a 403 gated error. In that case this
    helper prints how to unlock the official split and falls back to an ungated
    repack (`benjamin-paine/imagenet-1k-256x256`) that carries the same image and
    label schema with the standard 0 to 999 label ordering, so every experiment,
    including fragility, still works. Pass fallback_id=None to disable the
    fallback and see the raw error.
    """
    from datasets import load_dataset

    token = get_hf_token(token)
    def _load(ds_id):
        ds = load_dataset(ds_id, split=split, streaming=streaming, token=token)
        if shuffle:
            ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
        return ds

    def _probe(ds):
        # Force one real read. With streaming, load_dataset only fetches metadata
        # and the first file download (where a gated or fine-grained-token 403
        # actually happens) is deferred to iteration, otherwise deep inside a
        # DataLoader worker. Probing here surfaces the error where we can fall back.
        next(iter(ds))

    def _looks_gated(exc):
        msg = str(exc).lower()
        keys = ("gated", "not found", "403", "forbidden", "access", "permission", "token")
        return any(k in msg for k in keys)

    try:
        ds = _load(dataset_id)
        _probe(ds)
        print(f"loaded {dataset_id} [{split}]")
        return ds
    except Exception as exc:
        if not (fallback_id and _looks_gated(exc)):
            raise
        print(
            f"Could not read '{dataset_id}' with your token ({type(exc).__name__}).\n"
            f"To use the official split, do both of these:\n"
            f"  1. Accept the terms once at https://huggingface.co/datasets/{dataset_id}\n"
            f"     (click 'Agree and access repository').\n"
            f"  2. Use a token that can read gated repos: either a classic 'Read'\n"
            f"     token, or a fine-grained token with 'Read access to contents of\n"
            f"     all public gated repos you can access' enabled, at\n"
            f"     https://huggingface.co/settings/tokens\n"
            f"Falling back to the ungated mirror '{fallback_id}' for now "
            f"(same schema and label ordering, so every experiment still works)."
        )
        ds = _load(fallback_id)
        _probe(ds)
        print(f"loaded {fallback_id} [{split}]")
        return ds


def save_json(obj, path):
    ensure_dirs()
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path


def load_json(path):
    with open(path) as f:
        return json.load(f)


def summarize_curve(scores):
    """Compact summary of a per layer SSDC curve.

    peak        : max SSDC over depth.
    peak_layer  : depth at which the peak occurs.
    delta       : SSDC[1] - SSDC[0], the immediate recovery after the first block.
    decay       : peak - SSDC[last], how much SSDC falls from its peak by the end.
    final       : SSDC at the last layer.
    auc         : mean SSDC over depth.
    """
    s = np.asarray(scores, dtype=float)
    peak_layer = int(np.argmax(s))
    return {
        "peak": float(s.max()),
        "peak_layer": peak_layer,
        "delta": float(s[1] - s[0]) if s.size > 1 else 0.0,
        "decay": float(s.max() - s[-1]),
        "final": float(s[-1]),
        "auc": float(s.mean()),
    }


def summarize_ssdc(scores):
    """Return the standard curve summary together with JSON-safe scores."""
    summary = summarize_curve(scores)
    summary["scores"] = [float(score) for score in scores]
    return summary


def as_tensor(output):
    """Unwrap the primary tensor returned by a model component."""
    return output[0] if isinstance(output, tuple) else output


def forward_logits(model, source, pixel_values):
    """Run either supported ViT and return its classification logits."""
    if source == "transformers":
        return model(pixel_values=pixel_values).logits
    if source == "timm":
        return model(pixel_values)
    raise ValueError(f"unknown model source {source!r}")


def accuracy_counts(logits, labels):
    """Return integer correct and total counts for one batch."""
    predictions = logits.argmax(dim=-1).detach().cpu()
    labels = torch.as_tensor(labels).detach().cpu()
    return int((predictions == labels).sum().item()), int(labels.numel())


def model_device(model):
    return next(model.parameters()).device


def prepare_pixel_values(pixel_values, model, half=True):
    """Move a batch to the model device and match the experiment precision."""
    pixel_values = pixel_values.to(model_device(model))
    if half and pixel_values.device.type == "cuda":
        pixel_values = pixel_values.half()
    return pixel_values


def install_fixed_rpi_hook(model, source, permutation):
    """Install a deterministic patch-token permutation at the patch embedding."""
    from main.load_models import get_patch_embed_conv

    convolution = get_patch_embed_conv(model, source)

    def hook(module, inputs, output):
        batch, channels, height, width = output.shape
        flattened = output.reshape(batch, channels, -1).contiguous()
        index = permutation.to(device=output.device, dtype=torch.long)
        return flattened[:, :, index].reshape(batch, channels, height, width)

    return convolution.register_forward_hook(hook)


def patch_token_count(model, source, image_batch, half=True):
    """Infer the number of patch tokens from one preprocessed image batch."""
    from main.load_models import get_patch_embed_conv

    pixel_values = prepare_pixel_values(image_batch[:1], model, half=half)
    with torch.inference_mode():
        patch_grid = get_patch_embed_conv(model, source)(pixel_values)
    return int(patch_grid.shape[-2] * patch_grid.shape[-1])


def seeded_patch_permutation(num_patches, seed):
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randperm(int(num_patches), generator=generator)


def validate_layer_indices(layers, num_layers, name="layers", default=None):
    """Validate, deduplicate, and preserve the order of layer indices."""
    if layers is None:
        layers = default
    resolved = [int(layer) for layer in layers]
    if not resolved:
        raise ValueError(f"{name} must not be empty")
    unknown = [layer for layer in resolved if layer < 0 or layer >= num_layers]
    if unknown:
        raise ValueError(f"{name} {unknown} are out of range for a {num_layers}-layer model")
    return list(dict.fromkeys(resolved))


def collect_batches(
    dataset,
    processor,
    source,
    number_images,
    batch_size,
    corruption_type=None,
    severity=5,
):
    """Materialize a reusable image sample as CPU tensors.

    Causal interventions rerun the same images under several conditions. Keeping
    the preprocessed sample in memory avoids repeated streaming and decoding.
    """
    from main.prep_data import apply_corruption, get_corruption_registry

    corruption_fn = None
    if corruption_type:
        registry = get_corruption_registry()
        if corruption_type not in registry:
            raise ValueError(
                f"unknown corruption {corruption_type!r}; choices: {sorted(registry)}"
            )
        corruption_fn = registry[corruption_type]

    batches = []
    images = []
    labels = []
    collected = 0
    for item in dataset:
        image = item["image"].convert("RGB")
        if corruption_fn is not None:
            image = apply_corruption(image, corruption_fn, severity)
        if source == "transformers":
            tensor = processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        elif source == "timm":
            tensor = processor(image)
        else:
            raise ValueError(f"unknown model source {source!r}")
        images.append(tensor)
        labels.append(int(item["label"]))
        collected += 1

        if len(images) == batch_size:
            batches.append((torch.stack(images), torch.tensor(labels, dtype=torch.long)))
            images, labels = [], []
        if collected >= number_images:
            break

    if images:
        batches.append((torch.stack(images), torch.tensor(labels, dtype=torch.long)))
    return batches


def plot_curves(curves, title, ylabel="SSDC", xlabel="Layer", save_path=None, ax=None, styles=None):
    """Plot several named per layer curves on one axis.

    curves : dict name -> list of per layer values (all the same length).
    styles : optional dict name -> matplotlib kwargs.
    """
    import matplotlib.pyplot as plt

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    styles = styles or {}
    for name, values in curves.items():
        xs = list(range(len(values)))
        ax.plot(xs, values, marker="o", markersize=3, label=name, **styles.get(name, {}))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(0.0, color="0.7", linewidth=0.8, zorder=0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    if save_path and created:
        ensure_dirs()
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        print(f"saved {save_path}")
    return ax
