"""Unit tests for the modern SAE package (project_code/src/SAE).

These run on CPU with no GPU, no ImageNet, and no pretrained weights. The ViT
plumbing test builds a tiny randomly initialised ViT from a config, so it needs
no download. Run with:  python tests/test_sae.py
"""

import os
import sys

import torch

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "project_code", "src"))
sys.path.insert(0, SRC)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from SAE.sae import SAE, auxiliary_loss, _batch_topk, _topk_per_token
from SAE.activation_store import ActivationStore
from SAE.train import train_sae
from SAE.metrics import reconstruction_metrics, mean_max_cosine, downstream_delta


def check(name, cond):
    print(f"  {'ok ' if cond else 'FAIL'} {name}")
    assert cond, name


# --------------------------------------------------------------------------- #
# sparsity mechanics
# --------------------------------------------------------------------------- #
def test_topk_exact_l0():
    acts = torch.rand(7, 20)
    out = _topk_per_token(acts, 5)
    check("topk keeps exactly k per row", bool(((out > 0).sum(-1) == 5).all()))
    # kept values are the k largest
    check("topk keeps the largest", torch.allclose(out.sum(-1), acts.topk(5, -1).values.sum(-1)))


def test_batchtopk_budget():
    acts = torch.rand(8, 16)
    out = _batch_topk(acts, 4)  # 4 * 8 = 32 kept across the whole batch
    check("batchtopk keeps k*batch across the batch", int((out > 0).sum()) == 32)
    check("batchtopk average L0 is k", abs(float((out > 0).float().sum(-1).mean()) - 4) < 1e-6)


def test_b_dec_centering():
    mean = torch.arange(6).float()
    sae = SAE(6, 12, architecture="topk", k=2, mean=mean, use_b_dec=True)
    check("b_dec initialised to mean", torch.allclose(sae.b_dec.detach(), mean))
    z = torch.zeros(1, 12)
    check("decode(0) == b_dec", torch.allclose(sae.decode(z).squeeze(0), mean))


def test_decoder_unit_norm():
    sae = SAE(16, 64, architecture="batchtopk", k=8)
    with torch.no_grad():
        sae.W_dec.mul_(3.0)  # perturb norms
    sae.normalize_decoder()
    norms = sae.W_dec.norm(dim=1)
    check("decoder rows unit norm", torch.allclose(norms, torch.ones_like(norms), atol=1e-5))


def test_jumprelu_threshold_and_grad():
    torch.manual_seed(0)
    sae = SAE(8, 16, architecture="jumprelu", jumprelu_init_threshold=0.001)
    x = torch.randn(4, 8)
    x_hat, z, pre = sae(x)
    # push threshold very high -> nothing fires
    with torch.no_grad():
        sae.log_threshold.fill_(10.0)
    _, z_hi, _ = sae(x)
    check("high jumprelu threshold zeroes latents", int((z_hi > 0).sum()) == 0)
    # gradient reaches the threshold through the STE
    with torch.no_grad():
        sae.log_threshold.fill_(-2.0)
    _, z2, pre2 = sae(x)
    loss = (z2.sum())  # depends on the gate via the STE
    from SAE.sae import RectangleSTE
    theta = torch.exp(sae.log_threshold)
    gate = RectangleSTE.apply(pre2, theta, sae.jumprelu_bandwidth)
    gate.sum().backward()
    check("STE gives gradient to log_threshold", sae.log_threshold.grad is not None
          and bool(torch.isfinite(sae.log_threshold.grad).all()))


def test_aux_loss_uses_only_dead():
    sae = SAE(8, 32, architecture="batchtopk", k=4)
    x = torch.randn(10, 8)
    x_hat, z, pre = sae(x)
    dead = torch.zeros(32, dtype=torch.bool)
    dead[[3, 7, 11]] = True
    loss = auxiliary_loss(sae, x, x_hat, pre, dead, k_aux=8)
    check("aux loss is finite", bool(torch.isfinite(loss)))
    check("aux loss zero when no dead", float(auxiliary_loss(sae, x, x_hat, pre,
          torch.zeros(32, dtype=torch.bool), 8)) == 0.0)


# --------------------------------------------------------------------------- #
# store + training
# --------------------------------------------------------------------------- #
def test_store_normalization():
    x = torch.randn(2000, 32) * 5 + 3
    store = ActivationStore(x, normalize=True, seed=0)
    mean_norm = store.train.norm(dim=1).mean()
    import math
    check("store normalises to sqrt(d_model)", abs(float(mean_norm) - math.sqrt(32)) < 1.0)
    check("train/val disjoint sizes", store.n_train() + store.val.shape[0] == 2000)


def test_training_reduces_fvu():
    torch.manual_seed(0)
    # well conditioned sparse-dictionary data (overcomplete SAE, easy regime)
    D = torch.randn(32, 48); D = D / D.norm(dim=1, keepdim=True)
    mask = (torch.rand(6000, 32) < 0.15).float()
    z = mask * (-torch.log(torch.rand(6000, 32).clamp_min(1e-8)))
    x = z @ D + 0.03 * torch.randn(6000, 48)
    store = ActivationStore(x, seed=0)
    for arch, kw in [("batchtopk", dict(k=6)), ("jumprelu", dict(l0_coef=3e-3)),
                     ("relu_l1", dict(l1_coef=1e-3))]:
        sae, info = train_sae(store, d_hidden=96, n_epochs=40, batch_size=2048,
                              architecture=arch, **kw)
        check(f"{arch} reaches low FVU (<0.5)", info["final"]["fvu"] < 0.5)
        rec = mean_max_cosine(sae.W_dec.detach(), D)
        check(f"{arch} recovers true features (mmcs>0.8)", rec["recall_mmcs"] > 0.8)


# --------------------------------------------------------------------------- #
# real ViT plumbing (random weights, offline)
# --------------------------------------------------------------------------- #
def _tiny_vit():
    from transformers import ViTConfig, ViTForImageClassification
    cfg = ViTConfig(hidden_size=48, num_hidden_layers=3, num_attention_heads=4,
                    intermediate_size=96, image_size=32, patch_size=16, num_labels=10)
    return ViTForImageClassification(cfg).eval()


def test_extraction_shape_and_downstream_identity():
    from main.load_models import get_vit_blocks, num_prefix_tokens
    model = _tiny_vit()
    blocks = get_vit_blocks(model, "transformers")
    check("locates 3 blocks", len(blocks) == 3)
    n_prefix = num_prefix_tokens(model, "transformers")

    captured = []
    h = blocks[1].register_forward_hook(lambda m, i, o: captured.append(i[0].detach()))
    px = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        model(pixel_values=px)
    h.remove()
    act = captured[0]
    n_patches = (32 // 16) ** 2
    check("block input has cls + patches", act.shape == (2, n_patches + n_prefix, 48))

    # identity reconstruction must leave the model output unchanged
    class IdentitySAE:
        def __call__(self, x):
            return x, None, None
    d = downstream_delta(model, "transformers", blocks[1], IdentitySAE(), [px],
                         scale=0.5, device="cpu")
    check("identity splice keeps top-1 (agreement 1.0)", abs(d["top1_agreement"] - 1.0) < 1e-6)
    check("identity splice has ~zero KL", d["logit_kl"] < 1e-4)


ALL = [
    test_topk_exact_l0, test_batchtopk_budget, test_b_dec_centering,
    test_decoder_unit_norm, test_jumprelu_threshold_and_grad, test_aux_loss_uses_only_dead,
    test_store_normalization, test_training_reduces_fvu,
    test_extraction_shape_and_downstream_identity,
]

if __name__ == "__main__":
    for t in ALL:
        print(t.__name__)
        t()
    print("\nall SAE tests passed")
