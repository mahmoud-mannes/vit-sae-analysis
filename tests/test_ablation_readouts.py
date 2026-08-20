import os
import sys

import torch

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "project_code", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from experiments.ablation_sweep import build_specs, run_ablation_condition, run_final_readouts
from fake_vit import FakeImageTimm, approx, count_forward_hooks, make_image_batches


def test_run_final_readouts_defaults_to_final_layer():
    assert [spec["name"] for spec in build_specs(2)] == [
        "attn_L00",
        "attn_L01",
        "mlp_L00",
        "mlp_L01",
    ]
    torch.manual_seed(21)
    model = FakeImageTimm().eval()
    batches = make_image_batches(seed=22)

    default = run_final_readouts(model, "timm", batches, half=False)
    explicit = run_final_readouts(
        model,
        "timm",
        batches,
        half=False,
        ssdc_readout_layers=[1],
    )

    assert list(default["ssdc_by_layer"]) == [1]
    assert approx(default["final"], explicit["final"], 1e-12)
    assert approx(default["accuracy"], explicit["accuracy"], 1e-12)
    assert approx(default["ssdc_by_layer"][1], explicit["ssdc_by_layer"][1], 1e-12)


def test_ablation_readouts_capture_post_block_outputs():
    torch.manual_seed(31)
    model = FakeImageTimm().eval()
    batches = make_image_batches(seed=32)
    perm = torch.tensor([8, 3, 1, 7, 0, 5, 2, 6, 4])

    baseline = run_final_readouts(
        model,
        "timm",
        batches,
        half=False,
        ssdc_readout_layers=[0, 1],
    )
    ablated = run_ablation_condition(
        model,
        "timm",
        batches,
        perm,
        {"name": "attn_L01", "component": "attn", "layer": 1},
        half=False,
        ssdc_readout_layers=[0, 1],
    )["clean"]

    assert approx(ablated["ssdc_by_layer"][0], baseline["ssdc_by_layer"][0], 1e-12)
    assert not approx(ablated["ssdc_by_layer"][1], baseline["ssdc_by_layer"][1], 1e-6)
    assert approx(ablated["final"], ablated["ssdc_by_layer"][1], 1e-12)


def test_ablation_readout_hooks_do_not_leak():
    torch.manual_seed(41)
    model = FakeImageTimm().eval()
    batches = make_image_batches(seed=42)
    perm = torch.tensor([8, 3, 1, 7, 0, 5, 2, 6, 4])

    assert count_forward_hooks(model) == 0
    baseline_before = run_final_readouts(
        model,
        "timm",
        batches,
        perm=None,
        half=False,
        ssdc_readout_layers=[0],
    )
    assert count_forward_hooks(model) == 0

    run_ablation_condition(
        model,
        "timm",
        batches,
        perm,
        {"name": "attn_L00", "component": "attn", "layer": 0},
        half=False,
        ssdc_readout_layers=[0, 1],
    )
    assert count_forward_hooks(model) == 0

    baseline_after = run_final_readouts(
        model,
        "timm",
        batches,
        perm=None,
        half=False,
        ssdc_readout_layers=[0],
    )
    assert count_forward_hooks(model) == 0
    assert approx(baseline_before["final"], baseline_after["final"], 1e-12)
    assert approx(baseline_before["accuracy"], baseline_after["accuracy"], 1e-12)
    assert approx(
        baseline_before["ssdc_by_layer"][0],
        baseline_after["ssdc_by_layer"][0],
        1e-12,
    )


if __name__ == "__main__":
    tests = [value for key, value in sorted(globals().items()) if key.startswith("test_")]
    for test in tests:
        test()
    print(f"\nAll {len(tests)} tests passed.")
