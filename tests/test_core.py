"""Unit tests for the parts that do not need GPUs, ImageNet, or pretrained weights.

These validate the SSDC metric, the ablation hook contract (on tiny fake ViT
blocks that mimic the timm and transformers layouts), the effective rank metric,
and the light corruptions. Run with:  python tests/test_core.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "project_code", "src"))
sys.path.insert(0, SRC)

from metrics.ssdc import spatial_similarity_distance_correlation, _register_block_hooks
from metrics.effective_rank import batched_effective_rank, effective_rank_from_matrix
from interventions.ablation import AblationController, resolve_layers, num_blocks
from interventions.corruptions import gaussian_blur, jpeg_compression, pixelate, patch_shuffle


def approx(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


# --------------------------------------------------------------------------- #
# SSDC metric
# --------------------------------------------------------------------------- #
def test_ssdc_perfect_and_inverse():
    G = 4
    T = G * G
    coords = np.stack(
        np.meshgrid(np.arange(G), np.arange(G), indexing="ij"), axis=-1
    ).reshape(-1, 2)
    D = np.abs(coords[:, None, :] - coords[None, :, :]).sum(-1).astype(float)  # L1

    # If similarity equals negative distance, SSDC must be +1.
    S = -D
    assert approx(spatial_similarity_distance_correlation(S, G), 1.0, 1e-9)

    # If similarity equals distance, SSDC must be -1.
    assert approx(spatial_similarity_distance_correlation(D, G), -1.0, 1e-9)
    print("ok  test_ssdc_perfect_and_inverse")


def test_ssdc_random_near_zero():
    G = 6
    T = G * G
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(30):
        M = rng.standard_normal((T, T))
        S = (M + M.T) / 2
        vals.append(spatial_similarity_distance_correlation(S, G))
    assert abs(np.mean(vals)) < 0.05, np.mean(vals)
    print("ok  test_ssdc_random_near_zero")


def test_ssdc_grid_assert():
    try:
        spatial_similarity_distance_correlation(np.zeros((15, 15)), 4)
    except AssertionError:
        print("ok  test_ssdc_grid_assert")
        return
    raise AssertionError("expected grid_size**2 != T to assert")


# --------------------------------------------------------------------------- #
# Fake ViT blocks for the ablation contract
# --------------------------------------------------------------------------- #
class TimmBlock(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(C), nn.Linear(C, C)
        self.norm2, self.mlp = nn.LayerNorm(C), nn.Linear(C, C)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FakeTimm(nn.Module):
    def __init__(self, C, L):
        super().__init__()
        self.blocks = nn.ModuleList([TimmBlock(C) for _ in range(L)])
        self.num_prefix_tokens = 1

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class HFAttention(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.q = nn.Linear(C, C)

    def forward(self, hidden, head_mask=None, output_attentions=False):
        return (self.q(hidden),)  # transformers returns a tuple


class HFOutput(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.dense = nn.Linear(C, C)

    def forward(self, hidden_states, input_tensor):
        return self.dense(hidden_states) + input_tensor


class HFLayer(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.layernorm_before, self.attention = nn.LayerNorm(C), HFAttention(C)
        self.layernorm_after = nn.LayerNorm(C)
        self.intermediate = nn.Sequential(nn.Linear(C, C), nn.GELU())
        self.output = HFOutput(C)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        attn = self.attention(self.layernorm_before(hidden_states))[0]
        hidden_states = attn + hidden_states
        y = self.intermediate(self.layernorm_after(hidden_states))
        return (self.output(y, hidden_states),)


class _Encoder(nn.Module):
    def __init__(self, C, L):
        super().__init__()
        self.layer = nn.ModuleList([HFLayer(C) for _ in range(L)])


class _ViT(nn.Module):
    def __init__(self, C, L):
        super().__init__()
        self.encoder = _Encoder(C, L)


class FakeHF(nn.Module):
    def __init__(self, C, L):
        super().__init__()
        self.vit = _ViT(C, L)

    def forward(self, x):
        for lyr in self.vit.encoder.layer:
            x = lyr(x)[0]
        return x


def test_resolve_layers():
    assert resolve_layers([0, 1, 2], 12, "zero") == {0, 1, 2}
    assert resolve_layers([0, 1, 2], 12, "keep_only") == set(range(3, 12))
    try:
        resolve_layers([99], 12, "zero")
    except ValueError:
        print("ok  test_resolve_layers")
        return
    raise AssertionError("expected out of range layer to raise")


def test_ablation_all_makes_identity_timm():
    torch.manual_seed(0)
    model = FakeTimm(8, 4).eval()
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        assert num_blocks(model, "timm") == 4
        with AblationController(model, "timm", "attn", list(range(4))):
            with AblationController(model, "timm", "mlp", list(range(4))):
                out = model(x)
    assert torch.allclose(out, x, atol=1e-6), (out - x).abs().max().item()
    print("ok  test_ablation_all_makes_identity_timm")


def test_ablation_all_makes_identity_hf():
    torch.manual_seed(0)
    model = FakeHF(8, 3).eval()
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        assert num_blocks(model, "transformers") == 3
        with AblationController(model, "transformers", "attn", list(range(3))):
            with AblationController(model, "transformers", "mlp", list(range(3))):
                out = model(x)
    assert torch.allclose(out, x, atol=1e-6), (out - x).abs().max().item()
    print("ok  test_ablation_all_makes_identity_hf")


def test_ablation_only_target_layer_changes():
    # Ablating attention only at layer 1 must leave a 1 block model unchanged when
    # we ablate a non existent-effect layer, and must change output when it hits a
    # real layer. Here we check that attn ablation at layer 0 differs from baseline
    # and that keep_only of the same layer is the complement.
    torch.manual_seed(1)
    model = FakeTimm(8, 3).eval()
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        base = model(x)
        with AblationController(model, "timm", "attn", [0], mode="zero"):
            zeroed0 = model(x)
        with AblationController(model, "timm", "attn", [0], mode="keep_only"):
            kept0 = model(x)  # ablates layers 1 and 2, keeps 0
    assert not torch.allclose(base, zeroed0)
    assert not torch.allclose(base, kept0)
    assert not torch.allclose(zeroed0, kept0)
    print("ok  test_ablation_only_target_layer_changes")


def test_ssdc_capture_hook_shapes():
    model = FakeTimm(8, 4).eval()
    store = {"sum": {}, "count": {}}
    handles = _register_block_hooks(model, "timm", store)
    x = torch.randn(3, 17, 8)  # T=17 -> 16 patches -> 4x4 grid after dropping prefix
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    assert set(store["sum"].keys()) == {0, 1, 2, 3}
    for i in range(4):
        assert store["sum"][i].shape == (17, 17)
        assert store["count"][i] == 3
        mean_cos = (store["sum"][i] / store["count"][i]).numpy()
        assert approx(np.diag(mean_cos).mean(), 1.0, 1e-4)  # cosine self similarity
    print("ok  test_ssdc_capture_hook_shapes")


# --------------------------------------------------------------------------- #
# Effective rank
# --------------------------------------------------------------------------- #
def test_effective_rank_bounds():
    # Rank one matrix (all rows identical) -> effective rank ~ 1.
    v = torch.randn(10)
    rank1 = v.unsqueeze(0).repeat(6, 1).unsqueeze(0)  # [1, 6, 10]
    assert approx(batched_effective_rank(rank1)[0], 1.0, 1e-3), batched_effective_rank(rank1)[0]

    # Identity (orthonormal rows, equal singular values) -> effective rank = n.
    eye = torch.eye(5).unsqueeze(0)
    assert approx(effective_rank_from_matrix(eye[0]), 5.0, 1e-3), effective_rank_from_matrix(eye[0])
    print("ok  test_effective_rank_bounds")


# --------------------------------------------------------------------------- #
# Corruptions
# --------------------------------------------------------------------------- #
def test_corruptions_shapes():
    from PIL import Image

    img = Image.fromarray((np.random.rand(32, 40, 3) * 255).astype("uint8"))
    for fn in (gaussian_blur, jpeg_compression, pixelate, patch_shuffle):
        out = fn(img, severity=5)
        assert out.size == img.size, (fn.__name__, out.size, img.size)
        assert out.mode == "RGB"
    # Blur must actually reduce high frequency variance.
    blurred = np.array(gaussian_blur(img, 5)).astype(float)
    assert np.var(blurred) < np.var(np.array(img).astype(float))
    print("ok  test_corruptions_shapes")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed.")
