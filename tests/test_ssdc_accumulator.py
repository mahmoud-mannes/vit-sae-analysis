import os
import sys

import numpy as np
import torch
import torch.nn as nn

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "project_code", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import metrics.ssdc as ssdc_module
from metrics.ssdc import (
    SSDCAccumulator,
    _register_block_hooks,
    evaluate_ssdc,
    pairwise_cosine_similarity,
    spatial_similarity_distance_correlation,
)


def approx(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


class TimmBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Linear(channels, channels)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyTokenModel(nn.Module):
    def __init__(self, channels=6, layers=2):
        super().__init__()
        self.blocks = nn.ModuleList([TimmBlock(channels) for _ in range(layers)])
        self.num_prefix_tokens = 1

    def forward(self, tokens):
        for block in self.blocks:
            tokens = block(tokens)
        return tokens


def test_pairwise_cosine_similarity_shape_and_diagonal():
    torch.manual_seed(101)
    tokens = torch.randn(3, 9, 5)
    matrices = pairwise_cosine_similarity(tokens)

    assert matrices.shape == (3, 9, 9)
    assert torch.allclose(
        matrices.diagonal(dim1=-2, dim2=-1),
        torch.ones(3, 9),
        atol=1e-6,
    )


def test_ssdc_accumulator_averages_by_key():
    torch.manual_seed(202)
    first = torch.randn(2, 4, 3)
    second = torch.randn(1, 4, 3)
    other = torch.randn(3, 4, 3)
    key = ("layer0", "residual")

    accumulator = SSDCAccumulator()
    accumulator.add(key, first)
    accumulator.add(key, second)
    accumulator.add(("layer1", "residual"), other)

    first_cos = pairwise_cosine_similarity(first)
    second_cos = pairwise_cosine_similarity(second)
    expected = torch.cat([first_cos, second_cos], dim=0).mean(dim=0).numpy()

    actual = accumulator.mean_pairwise_matrix(key)
    assert np.allclose(actual, expected, atol=1e-6)
    assert set(accumulator.mean_pairwise_matrices()) == {
        ("layer0", "residual"),
        ("layer1", "residual"),
    }


def test_ssdc_accumulator_prefix_removal_and_ssdc():
    prefix = torch.tensor([[9.0, -4.0]], dtype=torch.float32)
    patches = torch.tensor(
        [[row, col] for row in range(3) for col in range(3)], dtype=torch.float32
    )
    tokens = torch.cat([prefix, patches], dim=0).unsqueeze(0)

    accumulator = SSDCAccumulator(n_prefix=1)
    accumulator.add("layer0", tokens)

    full = accumulator.mean_pairwise_matrix("layer0")
    trimmed = accumulator.mean_pairwise_matrix("layer0", remove_prefix=True)
    assert full.shape == (10, 10)
    assert trimmed.shape == (9, 9)
    assert np.allclose(trimmed, full[1:, 1:], atol=1e-6)

    expected = spatial_similarity_distance_correlation(trimmed, 3)
    assert approx(
        accumulator.ssdc("layer0", remove_prefix=True),
        expected,
        1e-9,
    )


def test_evaluate_ssdc_backward_compatible_outputs():
    torch.manual_seed(303)
    model = TinyTokenModel().eval()
    batches = [
        (torch.randn(2, 10, 6), torch.zeros(2, dtype=torch.long)),
        (torch.randn(1, 10, 6), torch.zeros(1, dtype=torch.long)),
    ]

    legacy_store = {"sum": {}, "count": {}}
    handles = _register_block_hooks(model, "timm", legacy_store)
    with torch.no_grad():
        for tokens, _ in batches:
            model(tokens)
    for handle in handles:
        handle.remove()

    expected_maps = []
    expected_scores = []
    for layer in sorted(legacy_store["sum"]):
        mean_cos = (legacy_store["sum"][layer] / legacy_store["count"][layer]).numpy()
        expected_maps.append(mean_cos)
        expected_scores.append(
            float(spatial_similarity_distance_correlation(mean_cos[1:, 1:], grid_size=3))
        )

    original_prep = ssdc_module.prep_data
    original_predict = ssdc_module.predict
    original_blocks = ssdc_module.get_vit_blocks
    original_attention = ssdc_module.get_block_attention
    try:
        ssdc_module.prep_data = lambda *args, **kwargs: batches
        ssdc_module.predict = lambda model, dataloader, source, RPI, magnitude, half=True: [
            model(tokens) for tokens, _ in dataloader
        ]
        ssdc_module.get_vit_blocks = lambda model, source: list(model.blocks)
        ssdc_module.get_block_attention = lambda block, source: block.attn

        scores, maps = evaluate_ssdc(
            model,
            processor=None,
            dataset=None,
            source="timm",
            number_images=3,
            batch_size=2,
            half=False,
            n_prefix=1,
        )
    finally:
        ssdc_module.prep_data = original_prep
        ssdc_module.predict = original_predict
        ssdc_module.get_vit_blocks = original_blocks
        ssdc_module.get_block_attention = original_attention

    assert len(scores) == len(expected_scores) == 2
    assert len(maps) == len(expected_maps) == 2
    for actual, expected in zip(scores, expected_scores):
        assert approx(actual, expected, 1e-9)
    for actual, expected in zip(maps, expected_maps):
        assert np.allclose(actual, expected, atol=1e-6)


if __name__ == "__main__":
    tests = [value for key, value in sorted(globals().items()) if key.startswith("test_")]
    for test in tests:
        test()
    print(f"All {len(tests)} tests passed.")
