"""Synthetic ground truth benchmark for the SAE variants.

Why synthetic. On real activations there is no ground truth dictionary, so you
can only measure reconstruction and sparsity, never whether the SAE recovered
the *right* features. Here we build data from a known sparse dictionary, so we
can measure recovery directly (mean max cosine similarity between the learned
and the true feature directions). This is the standard controlled test for
"is this SAE better" and it is fully reproducible offline.

Data model (mimics a residual stream): ``x = z D + b + noise`` where ``D`` is a
dictionary of unit norm feature directions, ``z`` is sparse and non negative
with heavy tailed magnitudes, ``b`` is a shared mean offset (so ``b_dec`` and
centring matter), and ``noise`` is small Gaussian.

Run:  python -m SAE.benchmark_synthetic
"""

from __future__ import annotations

import json
import os
import sys
import time
import torch

_SRC = os.path.abspath(os.path.dirname(__file__) + "/..")
if _SRC not in sys.path:
    sys.path.append(_SRC)

from SAE.activation_store import ActivationStore
from SAE.train import train_sae
from SAE.metrics import reconstruction_metrics, mean_max_cosine
from experiments.common import RESULTS_DIR, FIGURES_DIR, ensure_dirs


def make_synthetic_data(
    n_samples=40_000, d_model=256, n_true=1024, avg_active=10,
    noise=0.1, seed=0, device="cpu",
):
    """Superposition regime: n_true >> d_model, so features must share the space
    and the sparsity mechanism actually matters. This is where L1 shrinkage and
    feature absorption show up and where topk / batchtopk / jumprelu separate
    from plain L1."""
    g = torch.Generator().manual_seed(seed)
    D = torch.randn(n_true, d_model, generator=g)
    D = D / D.norm(dim=1, keepdim=True)                 # unit norm true features
    b = 0.3 * torch.randn(d_model, generator=g)          # shared mean offset

    # sparse, non negative codes with heavy tailed (exponential) magnitudes
    prob = avg_active / n_true
    mask = (torch.rand(n_samples, n_true, generator=g) < prob).float()
    mags = -torch.log(torch.rand(n_samples, n_true, generator=g).clamp_min(1e-8))
    z = mask * mags
    x = z @ D + b + noise * torch.randn(n_samples, d_model, generator=g)
    return x.to(device), D.to(device)


def run(
    d_hidden=1024, n_epochs=15, batch_size=4096, out_name="synthetic",
    device="cpu", seed=0,
):
    ensure_dirs()
    t0 = time.time()
    print("generating synthetic data...", flush=True)
    x, D_true = make_synthetic_data(seed=seed, device=device)
    store = ActivationStore(x, val_fraction=0.05, normalize=True, seed=seed)
    print(f"  train tokens {store.n_train()}, d_model {store.d_model}, "
          f"true features {D_true.shape[0]}", flush=True)

    # (label, kwargs) for each configuration. b_dec is on for every modern
    # variant; the baseline reproduces the old repo recipe (L1, no b_dec).
    configs = []
    for c in [0.3, 3.0, 30.0]:
        configs.append((f"baseline_L1_no_bdec|c={c:g}",
                        dict(architecture="relu_l1", l1_coef=c, use_b_dec=False)))
    for c in [3.0]:
        configs.append((f"L1_bdec|c={c:g}",
                        dict(architecture="relu_l1", l1_coef=c, use_b_dec=True)))
    for k in [6, 10, 16]:
        configs.append((f"TopK|k={k}", dict(architecture="topk", k=k)))
    for k in [6, 10, 16]:
        configs.append((f"BatchTopK|k={k}", dict(architecture="batchtopk", k=k)))
    for c in [0.3, 1.0, 3.0]:
        configs.append((f"JumpReLU|c={c:g}",
                        dict(architecture="jumprelu", l0_coef=c)))

    rows = []
    for label, kw in configs:
        family = label.split("|")[0]
        t = time.time()
        sae, info = train_sae(
            store, d_hidden=d_hidden, n_epochs=n_epochs, batch_size=batch_size,
            dead_tokens_threshold=50_000, aux_coef=0.125, k_aux=256,
            seed=seed, device=device, **kw,
        )
        m = info["final"]
        rec = mean_max_cosine(sae.W_dec.detach(), D_true)
        row = {
            "label": label, "family": family,
            "fvu": m["fvu"], "l0": m["l0"], "dead_fraction": m["dead_fraction"],
            "cosine": m["cosine"],
            "recall_mmcs": rec["recall_mmcs"], "precision_mmcs": rec["precision_mmcs"],
        }
        rows.append(row)
        print(f"{label:28s} L0 {m['l0']:6.1f}  FVU {m['fvu']:.4f}  "
              f"dead {m['dead_fraction']:.3f}  recovery {rec['recall_mmcs']:.3f}  "
              f"({time.time()-t:.0f}s)", flush=True)

    out_dir = os.path.join(RESULTS_DIR, "runs", out_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sae_benchmark.json"), "w") as f:
        json.dump({"config": {"d_hidden": d_hidden, "n_epochs": n_epochs,
                              "n_true": int(D_true.shape[0])}, "rows": rows}, f, indent=2)
    _plot(rows, os.path.join(FIGURES_DIR, f"sae_{out_name}_frontier.png"))
    print(f"\ndone in {time.time()-t0:.0f}s. results -> {out_dir}", flush=True)
    return rows


def _plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = {}
    for r in rows:
        families.setdefault(r["family"], []).append(r)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for fam, rs in families.items():
        rs = sorted(rs, key=lambda r: r["l0"])
        l0 = [r["l0"] for r in rs]
        ax[0].plot(l0, [r["fvu"] for r in rs], "o-", label=fam)
        ax[1].plot(l0, [r["recall_mmcs"] for r in rs], "o-", label=fam)
    ax[0].set_xlabel("L0 (active latents / token)"); ax[0].set_ylabel("FVU (lower better)")
    ax[0].set_title("Sparsity vs fidelity frontier"); ax[0].set_xscale("log"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("L0 (active latents / token)"); ax[1].set_ylabel("ground truth recovery (higher better)")
    ax[1].set_title("Feature recovery (mean max cosine)"); ax[1].set_xscale("log"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"figure -> {path}", flush=True)


if __name__ == "__main__":
    run()
