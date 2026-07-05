"""Shared helpers for the experiment scripts: paths, dataset streaming, small
curve summaries, JSON IO, and plotting.

Everything here is deliberately thin so the scripts and the Colab notebook can
share the same building blocks.
"""

import json
import os
import sys

import numpy as np

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


def load_imagenet(split="validation", streaming=True, token=None, shuffle=False, seed=0, buffer_size=2000):
    """Stream ImageNet-1k from the Hub. The split is gated, so a token is needed.

    Returns a (streaming) HuggingFace dataset yielding {'image': PIL, 'label': int}.
    """
    from datasets import load_dataset

    token = get_hf_token(token)
    ds = load_dataset("ILSVRC/imagenet-1k", split=split, streaming=streaming, token=token)
    if shuffle:
        ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
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
