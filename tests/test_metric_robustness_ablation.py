import os
import sys

import numpy as np
import torch

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "project_code", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from experiments.metric_robustness_ablation import (
    parse_specs,
    run_ablation_condition,
    run_multi_metric_readouts,
)
from metrics.ssdc import (
    pairwise_similarity_matrices,
    spatial_similarity_distance_correlation,
)
from fake_vit import FakeImageTimm, count_forward_hooks, make_image_batches


def test_pairwise_metrics_have_expected_translation_behavior():
    torch.manual_seed(51)
    tokens = torch.randn(3, 9, 5)
    translated = tokens + torch.tensor([[[7.0, -2.0, 4.0, 1.0, -5.0]]])
    original = pairwise_similarity_matrices(tokens)
    shifted = pairwise_similarity_matrices(translated)

    assert not torch.allclose(original["cosine"], shifted["cosine"], atol=1e-5)
    assert torch.allclose(
        original["centered_cosine"], shifted["centered_cosine"], atol=2e-5
    )
    assert torch.allclose(
        original["negative_euclidean"], shifted["negative_euclidean"], atol=2e-5
    )


def test_all_metrics_detect_a_synthetic_spatial_grid():
    coords = torch.tensor(
        [[row, col] for row in range(3) for col in range(3)], dtype=torch.float32
    ).unsqueeze(0)
    matrices = pairwise_similarity_matrices(coords)

    scores = {
        name: spatial_similarity_distance_correlation(matrix[0].numpy(), 3)
        for name, matrix in matrices.items()
    }
    assert scores["cosine"] > 0.25
    assert scores["centered_cosine"] > 0.25
    assert scores["negative_euclidean"] > 0.85


def test_centering_excludes_the_prefix_token():
    torch.manual_seed(53)
    model = FakeImageTimm(grid_size=3).eval()
    batches = make_image_batches(seed=54, grid_size=3)

    result = run_multi_metric_readouts(
        model, "timm", batches, readout_layers=[0], half=False
    )

    captured = []

    def capture(module, inputs, output):
        captured.append(output.detach()[:, 1:])

    handle = model.blocks[0].register_forward_hook(capture)
    with torch.inference_mode():
        model(batches[0][0])
    handle.remove()
    centered = pairwise_similarity_matrices(
        captured[0], metrics=("centered_cosine",)
    )["centered_cosine"].mean(dim=0).numpy()
    expected = spatial_similarity_distance_correlation(centered, grid_size=3)

    assert np.isclose(
        result["metrics_by_layer"][0]["centered_cosine"], expected, atol=1e-12
    )


def test_spec_parser_validates_component_and_layer():
    assert parse_specs(["attn_L03", "mlp_L4"], 12) == [
        {"name": "attn_L03", "component": "attn", "layer": 3},
        {"name": "mlp_L04", "component": "mlp", "layer": 4},
    ]
    for bad_spec in ("head_L03", "attn_3", "attn_L12"):
        try:
            parse_specs([bad_spec], 12)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected {bad_spec!r} to fail")


def test_multi_metric_readouts_and_ablation_leave_no_hooks():
    torch.manual_seed(61)
    model = FakeImageTimm().eval()
    batches = make_image_batches(seed=62)
    perm = torch.tensor([8, 3, 1, 7, 0, 5, 2, 6, 4])

    baseline = run_multi_metric_readouts(
        model, "timm", batches, readout_layers=[0, 1], half=False
    )
    assert count_forward_hooks(model) == 0
    assert set(baseline["metrics_by_layer"]) == {0, 1}
    assert all(
        np.isfinite(score)
        for layer_scores in baseline["metrics_by_layer"].values()
        for score in layer_scores.values()
    )

    ablated = run_ablation_condition(
        model,
        "timm",
        batches,
        perm,
        {"name": "attn_L01", "component": "attn", "layer": 1},
        readout_layers=[0, 1],
        half=False,
    )["clean"]
    assert count_forward_hooks(model) == 0
    for metric_name in baseline["metrics_by_layer"][0]:
        assert np.isclose(
            ablated["metrics_by_layer"][0][metric_name],
            baseline["metrics_by_layer"][0][metric_name],
            atol=1e-12,
        )


if __name__ == "__main__":
    tests = [value for key, value in sorted(globals().items()) if key.startswith("test_")]
    for test in tests:
        test()
    print(f"All {len(tests)} tests passed.")
