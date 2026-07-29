"""Train and compare SAE variants on real ViT residual stream activations.

This is the on domain counterpart to ``benchmark_synthetic.py``. It streams
ImageNet images, captures the residual stream entering one block of the
pretrained ViT-Base, and trains every SAE variant on those activations. It
reports the held out reconstruction frontier (FVU vs L0, dead fraction, cosine)
and, for the current baseline and the recommended BatchTopK model, the
downstream faithfulness of a reconstruction splice (top-1 agreement and logit KL
when the model reads the SAE reconstruction instead of the true activation).

There is no ground truth dictionary here, so recovery is not measurable; that is
what the synthetic benchmark is for. What this shows is that the modern recipe
also wins on real activations and preserves what the model computes.

Run (needs an HF token in the environment for the gated split; it falls back to
an ungated ImageNet repack otherwise):

    python -m SAE.run_real --images 512 --layer 6 --epochs 10
"""

from __future__ import annotations

import argparse
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
from SAE.metrics import reconstruction_metrics, downstream_delta
from main.load_models import load_model, get_vit_blocks, num_prefix_tokens
from experiments.common import load_imagenet, get_hf_token, RESULTS_DIR, FIGURES_DIR, ensure_dirs


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


@torch.no_grad()
def extract(kind, layer, images, batch_size, n_downstream, device):
    model, processor, source = load_model(kind, device=device)
    blocks = get_vit_blocks(model, source)
    n_prefix = num_prefix_tokens(model, source)
    cap = []
    h = blocks[layer].register_forward_hook(lambda mod, i, o: cap.append(i[0].detach()))

    ds = load_imagenet(streaming=True, token=get_hf_token())
    it = iter(ds)
    chunks, pixel_batches, seen, buf = [], [], 0, []

    def flush(imgs):
        if source == "transformers":
            px = torch.cat([processor(images=im, return_tensors="pt")["pixel_values"] for im in imgs], 0)
        else:
            px = torch.stack([processor(im) for im in imgs], 0)
        model(pixel_values=px.to(device)) if source == "transformers" else model(px.to(device))
        act = cap.pop()[:, n_prefix:, :]
        chunks.append(act.reshape(-1, act.shape[-1]).float().cpu())
        if len(pixel_batches) * batch_size < n_downstream:
            pixel_batches.append(px.cpu())

    while seen < images:
        try:
            item = next(it)
        except StopIteration:
            break
        buf.append(item["image"].convert("RGB"))
        seen += 1
        if len(buf) == batch_size:
            flush(buf); buf = []
            if seen % (batch_size * 4) == 0:
                log(f"  extracted {seen}/{images}")
    if buf:
        flush(buf)
    h.remove()
    acts = torch.cat(chunks, 0).contiguous()
    log(f"activation store {tuple(acts.shape)} from {seen} imgs, layer {layer}")
    return acts, model, source, blocks[layer], pixel_batches, n_prefix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="ape")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--images", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--d-hidden", type=int, default=1536)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--train-batch", type=int, default=8192)
    ap.add_argument("--n-downstream", type=int, default=128)
    ap.add_argument("--out", default="imagenet1k_val")
    args = ap.parse_args()

    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    log(f"extracting real ViT ({args.kind}) activations...")
    acts, model, source, block, pixel_batches, n_prefix = extract(
        args.kind, args.layer, args.images, args.batch_size, args.n_downstream, device
    )
    store = ActivationStore(acts, val_fraction=0.05, normalize=True, seed=0)
    log(f"store scale {store.scale:.4f}; train {store.n_train()} tokens; d_model {store.d_model}")

    # AuxK settings tuned to keep dead latents low at the sparse operating point.
    tune = dict(dead_tokens_threshold=20_000, aux_coef=0.125, k_aux=512)

    configs = [
        ("baseline_L1_no_bdec|c=1",  dict(architecture="relu_l1", l1_coef=1.0, use_b_dec=False)),
        ("baseline_L1_no_bdec|c=8",  dict(architecture="relu_l1", l1_coef=8.0, use_b_dec=False)),
        ("baseline_L1_no_bdec|c=40", dict(architecture="relu_l1", l1_coef=40.0, use_b_dec=False)),
        ("L1_bdec|c=8",              dict(architecture="relu_l1", l1_coef=8.0, use_b_dec=True)),
        ("TopK|k=32",                dict(architecture="topk", k=32)),
        ("BatchTopK|k=16",           dict(architecture="batchtopk", k=16)),
        ("BatchTopK|k=32",           dict(architecture="batchtopk", k=32)),
        ("BatchTopK|k=64",           dict(architecture="batchtopk", k=64)),
        ("JumpReLU|c=0.3",           dict(architecture="jumprelu", l0_coef=0.3)),
        ("JumpReLU|c=1.0",           dict(architecture="jumprelu", l0_coef=1.0)),
    ]
    # downstream faithfulness is measured for the current baseline recipe and the
    # sparse modern models, at matched L0 where possible, to bound forward passes.
    downstream_for = {"baseline_L1_no_bdec|c=1", "TopK|k=32", "BatchTopK|k=32"}

    rows, saes = [], {}
    for label, kw in configs:
        t = time.time()
        sae, info = train_sae(
            store, d_hidden=args.d_hidden, n_epochs=args.epochs,
            batch_size=args.train_batch, device=device, seed=0, **tune, **kw,
        )
        m = info["final"]
        row = {"label": label, "family": label.split("|")[0],
               "fvu": m["fvu"], "l0": m["l0"], "dead_fraction": m["dead_fraction"],
               "cosine": m["cosine"]}
        if label in downstream_for:
            d = downstream_delta(model, source, block, sae, pixel_batches,
                                 scale=store.scale, n_prefix=n_prefix, device=device)
            row.update(down_top1_agreement=d["top1_agreement"], down_logit_kl=d["logit_kl"])
        rows.append(row)
        saes[label] = sae
        extra = (f"  down_agree {row.get('down_top1_agreement', float('nan')):.3f}"
                 if label in downstream_for else "")
        log(f"{label:26s} L0 {m['l0']:6.1f}  FVU {m['fvu']:.4f}  dead {m['dead_fraction']:.3f}"
            f"  cos {m['cosine']:.3f}{extra}  ({time.time()-t:.0f}s)")

    out_dir = os.path.join(RESULTS_DIR, "runs", args.out)
    os.makedirs(out_dir, exist_ok=True)
    meta = {"kind": args.kind, "layer": args.layer, "images": args.images,
            "d_model": store.d_model, "d_hidden": args.d_hidden, "epochs": args.epochs,
            "n_tokens": store.n_train() + store.val.shape[0], "scale": store.scale}
    with open(os.path.join(out_dir, "sae_real_benchmark.json"), "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=2)
    _plot(rows, os.path.join(FIGURES_DIR, "sae_real_frontier.png"), meta)
    log(f"DONE in {time.time()-t0:.0f}s -> {out_dir}/sae_real_benchmark.json")


def _plot(rows, path, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams = {}
    for r in rows:
        fams.setdefault(r["family"], []).append(r)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for fam, rs in fams.items():
        rs = sorted(rs, key=lambda r: r["l0"])
        ax.plot([r["l0"] for r in rs], [r["fvu"] for r in rs], "o-", label=fam)
    ax.set_xlabel("L0 (active latents / token)")
    ax.set_ylabel("FVU (lower is better)")
    ax.set_title(f"Real ViT-{meta['kind'].upper()} layer {meta['layer']}: sparsity vs fidelity")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    log(f"figure -> {path}")


if __name__ == "__main__":
    main()
