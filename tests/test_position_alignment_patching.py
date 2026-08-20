import os
import sys

import torch

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "project_code", "src"))
sys.path.insert(0, SRC)

from experiments.position_alignment_patching import (
    CONTROL_ALIGNED,
    CONTROL_DIFFERENT_IMAGE,
    CONTROL_MEAN_DONOR,
    CONTROL_MISALIGNED,
    _apply_fixed_cls_permutation,
    _build_batch_derangement,
    _build_patch_token_permutation,
    _mean_donor,
    _control_contrasts,
    _readout_scores,
    _transform_cached_activation,
    run_control_condition,
)
from experiments.activation_patching import run_baseline
from fake_vit import FakeImageTimm, approx


def test_fixed_cls_permutation_keeps_cls_token_for_token_major_tensors():
    value = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    spec = {"component": "residual"}
    perm = torch.tensor([2, 0, 3, 1])

    transformed = _apply_fixed_cls_permutation(value, spec, perm)

    assert torch.equal(transformed[:, 0], value[:, 0])
    assert torch.equal(transformed[:, 1], value[:, 3])
    assert torch.equal(transformed[:, 2], value[:, 1])
    assert torch.equal(transformed[:, 3], value[:, 4])
    assert torch.equal(transformed[:, 4], value[:, 2])


def test_fixed_cls_permutation_keeps_cls_token_for_timm_qkv_layout():
    value = torch.arange(2 * 3 * 5 * 2, dtype=torch.float32).reshape(2, 3, 5, 2)
    spec = {"component": "qkv", "n_heads": 3}
    perm = torch.tensor([1, 3, 0, 2])

    transformed = _apply_fixed_cls_permutation(value, spec, perm)

    assert torch.equal(transformed[:, :, 0], value[:, :, 0])
    assert torch.equal(transformed[:, :, 1], value[:, :, 2])
    assert torch.equal(transformed[:, :, 2], value[:, :, 4])
    assert torch.equal(transformed[:, :, 3], value[:, :, 1])
    assert torch.equal(transformed[:, :, 4], value[:, :, 3])


def test_deterministic_batch_derangement_has_no_fixed_points():
    first = _build_batch_derangement(5, seed=7)
    second = _build_batch_derangement(5, seed=7)

    assert torch.equal(first, second)
    assert not torch.any(first == torch.arange(5))


def test_different_image_control_requires_batch_size_above_one():
    try:
        _build_batch_derangement(1, seed=0)
    except ValueError as exc:
        assert "batch size > 1" in str(exc)
        return
    raise AssertionError("expected a batch-size guard for different-image control")


def test_readout_scores_only_return_selected_layers():
    scores = [0.1, 0.2, 0.3, 0.4]
    assert _readout_scores(scores, [1, 3]) == {"1": 0.2, "3": 0.4}


def test_aligned_same_image_residual_patch_matches_clean_readout():
    torch.manual_seed(11)
    model = FakeImageTimm(grid_size=2).eval()
    images = torch.randn(6, 3, 2, 2)
    labels = torch.zeros(6, dtype=torch.long)
    batches = [(images, labels)]
    rpi_perm = torch.tensor([3, 1, 0, 2])

    clean = run_baseline(model, "timm", batches, perm=None, half=False)
    aligned = run_control_condition(
        model,
        "timm",
        batches,
        rpi_perm,
        {"name": "residual_L01", "component": "residual", "layer": 1},
        CONTROL_ALIGNED,
        readout_layers=[1],
        half=False,
    )

    assert approx(aligned["readout_layers"]["1"], clean["scores"][1], 1e-12)


def test_control_transforms_preserve_expected_axes():
    value = torch.arange(3 * 5 * 2, dtype=torch.float32).reshape(3, 5, 2)
    perm = _build_patch_token_permutation(4, seed=3)
    derangement = _build_batch_derangement(3, seed=2)
    spec = {"component": "attn"}

    aligned = _transform_cached_activation(value, spec, CONTROL_ALIGNED)
    misaligned = _transform_cached_activation(
        value,
        spec,
        CONTROL_MISALIGNED,
        patch_permutation=perm,
    )
    deranged = _transform_cached_activation(
        value,
        spec,
        CONTROL_DIFFERENT_IMAGE,
        batch_derangement=derangement,
    )

    assert torch.equal(aligned, value)
    assert torch.equal(misaligned[:, 0], value[:, 0])
    assert torch.equal(deranged, value.index_select(0, derangement))


def test_mean_donor_broadcasts_batch_average():
    value = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3, 4, 2)

    donor = _mean_donor(value)

    expected = value.mean(dim=0, keepdim=True).expand_as(value)
    assert torch.equal(donor, expected)


def test_control_contrasts_separate_position_image_and_accuracy_effects():
    spec_results = {
        CONTROL_ALIGNED: {"readout_layers": {"4": 0.7, "5": 0.6}, "accuracy": 0.1},
        CONTROL_MISALIGNED: {"readout_layers": {"4": 0.4, "5": 0.5}, "accuracy": 0.9},
        CONTROL_DIFFERENT_IMAGE: {"readout_layers": {"4": 0.2, "5": 0.3}, "accuracy": 0.8},
        CONTROL_MEAN_DONOR: {"readout_layers": {"4": 0.25, "5": 0.35}, "accuracy": 0.7},
    }

    contrasts = _control_contrasts(spec_results, [4, 5])

    assert contrasts["4"]["aligned_gt_misaligned"] is True
    assert contrasts["4"]["different_image_gt_misaligned"] is False
    assert approx(contrasts["5"]["aligned_minus_misaligned"], 0.1)
    assert approx(contrasts["5"]["aligned_minus_different_image"], 0.3)
    assert approx(contrasts["5"]["mean_donor_minus_misaligned"], -0.15)
    assert approx(contrasts["accuracy"]["aligned_minus_different_image"], -0.7)


if __name__ == "__main__":
    tests = [value for key, value in sorted(globals().items()) if key.startswith("test_")]
    for test in tests:
        test()
    print(f"All {len(tests)} tests passed.")
