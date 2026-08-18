import os
import sys

import numpy as np
import torch
from PIL import Image

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "project_code", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from experiments import common
from fake_vit import FakeImageTimm


def test_validate_layer_indices_preserves_order_and_removes_duplicates():
    assert common.validate_layer_indices([3, 1, 3], 4) == [3, 1]
    try:
        common.validate_layer_indices([4], 4)
    except ValueError as error:
        assert "out of range" in str(error)
    else:
        raise AssertionError("expected an out-of-range layer to fail")


def test_collect_batches_respects_the_exact_image_limit():
    image = Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8))
    dataset = [{"image": image, "label": index % 2} for index in range(8)]
    processor = lambda _image: torch.zeros(3, 2, 2)

    batches = common.collect_batches(
        dataset,
        processor,
        source="timm",
        number_images=5,
        batch_size=3,
    )

    assert [labels.numel() for _, labels in batches] == [3, 2]
    assert sum(labels.numel() for _, labels in batches) == 5


def test_patch_count_and_seeded_permutation_are_deterministic():
    torch.manual_seed(71)
    model = FakeImageTimm(grid_size=2).eval()
    images = torch.randn(2, 3, 2, 2)

    assert common.patch_token_count(model, "timm", images, half=False) == 4
    first = common.seeded_patch_permutation(4, seed=9)
    second = common.seeded_patch_permutation(4, seed=9)
    assert torch.equal(first, second)
    assert torch.equal(torch.sort(first).values, torch.arange(4))


if __name__ == "__main__":
    tests = [value for key, value in sorted(globals().items()) if key.startswith("test_")]
    for test in tests:
        test()
    print(f"All {len(tests)} tests passed.")
