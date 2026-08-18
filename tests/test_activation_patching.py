import os
import sys

import torch

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "project_code", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from experiments.activation_patching import (
    install_clean_cache_hook,
    run_baseline,
    run_patch_condition,
)
from fake_vit import FakeImageTimm, approx, make_image_batches


def test_ssdc_excludes_the_clean_cache_pass():
    torch.manual_seed(7)
    model = FakeImageTimm(grid_size=2).eval()
    batches = make_image_batches(seed=8, grid_size=2)
    permutation = torch.tensor([2, 0, 3, 1])

    rpi = run_baseline(model, "timm", batches, perm=permutation, half=False)
    patched = run_patch_condition(
        model,
        "timm",
        batches,
        permutation,
        {"name": "attn_L01", "component": "attn", "layer": 1},
        half=False,
    )

    assert approx(patched["scores"][0], rpi["scores"][0], 1e-12)


def test_residual_patch_is_visible_to_the_ssdc_readout():
    torch.manual_seed(11)
    model = FakeImageTimm(grid_size=2).eval()
    batches = make_image_batches(seed=12, grid_size=2)
    permutation = torch.tensor([3, 1, 0, 2])

    clean = run_baseline(model, "timm", batches, perm=None, half=False)
    patched = run_patch_condition(
        model,
        "timm",
        batches,
        permutation,
        {"name": "residual_L01", "component": "residual", "layer": 1},
        half=False,
    )

    assert approx(patched["scores"][1], clean["scores"][1], 1e-12)


def test_full_precision_cache_preserves_activation_dtype():
    torch.manual_seed(15)
    model = FakeImageTimm(grid_size=2).eval()
    images = make_image_batches(seed=16, grid_size=2)[0][0]
    cache = {}

    handle = install_clean_cache_hook(
        model,
        "timm",
        {"name": "residual_L01", "component": "residual", "layer": 1},
        cache,
    )
    with torch.inference_mode():
        model(images)
    handle.remove()

    assert cache["value"].dtype == torch.float32


if __name__ == "__main__":
    tests = [value for key, value in sorted(globals().items()) if key.startswith("test_")]
    for test in tests:
        test()
    print(f"All {len(tests)} tests passed.")
