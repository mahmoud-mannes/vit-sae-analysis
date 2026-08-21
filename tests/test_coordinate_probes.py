import os
import sys

import numpy as np
import torch

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "project_code", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from metrics.position_probe import (
    evaluate_coordinate_probe,
    fit_shared_ridge_probes,
    fit_shared_ridge_probes_torch,
    normalized_square_grid_coordinates,
    split_image_indices,
)


def test_normalized_square_grid_coordinates_returns_row_and_column_targets():
    rows, cols = normalized_square_grid_coordinates(4)

    assert np.allclose(rows, np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32))
    assert np.allclose(cols, np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32))


def test_split_image_indices_is_deterministic_and_image_level():
    split = split_image_indices(10, seed=7)
    split_repeat = split_image_indices(10, seed=7)

    assert set(split) == {"train", "val", "test"}
    for key in split:
        assert np.array_equal(split[key], split_repeat[key])

    train = set(split["train"].tolist())
    val = set(split["val"].tolist())
    test = set(split["test"].tolist())

    assert train
    assert val
    assert test
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(range(10))


def test_coordinate_probe_beats_independently_shuffled_null_targets():
    num_images = 8
    rows = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    cols = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    image_ids = np.arange(num_images, dtype=np.float32)

    train_outputs = []
    eval_outputs = []
    for image_id in image_ids:
        train_tokens = np.stack(
            [
                rows,
                cols,
                np.full_like(rows, 0.01 * image_id),
                np.full_like(rows, 1.0),
            ],
            axis=-1,
        )
        eval_tokens = np.stack(
            [
                rows,
                cols,
                np.full_like(rows, -0.02 * image_id),
                np.full_like(rows, -3.0),
            ],
            axis=-1,
        )
        train_outputs.append(train_tokens)
        eval_outputs.append(eval_tokens)
    train_outputs = np.asarray(train_outputs, dtype=np.float32)
    eval_outputs = np.asarray(eval_outputs, dtype=np.float32)

    split = split_image_indices(num_images, seed=3)
    metrics = evaluate_coordinate_probe(
        train_outputs,
        eval_outputs,
        split,
        shuffle_seed=17,
    )
    metrics_repeat = evaluate_coordinate_probe(
        train_outputs,
        eval_outputs,
        split,
        shuffle_seed=17,
    )

    assert metrics["row"]["r2"] > 0.98
    assert metrics["column"]["r2"] > 0.98
    assert metrics["negative_control"]["row"]["r2"] < 0.1
    assert metrics["negative_control"]["column"]["r2"] < 0.1
    assert metrics["negative_control"] == metrics_repeat["negative_control"]


def test_torch_ridge_matches_numpy_ridge_on_cpu_and_cuda():
    rng = np.random.default_rng(23)
    train_x = rng.normal(size=(40, 12)).astype(np.float32)
    val_x = rng.normal(size=(16, 12)).astype(np.float32)
    test_x = rng.normal(size=(18, 12)).astype(np.float32)
    weights = rng.normal(size=12)

    def target(values):
        return values @ weights + 0.02 * rng.normal(size=values.shape[0])

    targets = {
        "position": (target(train_x), target(val_x), target(test_x)),
    }
    numpy_result = fit_shared_ridge_probes(
        train_x, val_x, test_x, targets, alpha_grid=(1e-2, 1e-1, 1.0)
    )
    cpu_result = fit_shared_ridge_probes_torch(
        train_x,
        val_x,
        test_x,
        targets,
        alpha_grid=(1e-2, 1e-1, 1.0),
        device="cpu",
    )

    assert cpu_result["position"]["alpha"] == numpy_result["position"]["alpha"]
    assert abs(cpu_result["position"]["r2"] - numpy_result["position"]["r2"]) < 1e-4

    default_result = fit_shared_ridge_probes_torch(
        train_x,
        val_x,
        test_x,
        targets,
        alpha_grid=(1e-2, 1e-1, 1.0),
    )
    assert default_result["position"]["alpha"] == numpy_result["position"]["alpha"]
    assert abs(default_result["position"]["r2"] - numpy_result["position"]["r2"]) < 1e-4

    if torch.cuda.is_available():
        cuda_result = fit_shared_ridge_probes_torch(
            train_x,
            val_x,
            test_x,
            targets,
            alpha_grid=(1e-2, 1e-1, 1.0),
            device="cuda",
        )
        assert cuda_result["position"]["alpha"] == numpy_result["position"]["alpha"]
        assert abs(cuda_result["position"]["r2"] - numpy_result["position"]["r2"]) < 1e-4


def test_ridge_requires_at_least_one_alpha():
    features = np.arange(12, dtype=np.float32).reshape(6, 2)
    targets = np.arange(6, dtype=np.float32)
    try:
        fit_shared_ridge_probes(
            features[:3],
            features[3:4],
            features[4:],
            {"position": (targets[:3], targets[3:4], targets[4:])},
            alpha_grid=(),
        )
    except ValueError as error:
        assert "alpha_grid" in str(error)
    else:
        raise AssertionError("expected an empty alpha grid to fail")


if __name__ == "__main__":
    tests = [value for key, value in sorted(globals().items()) if key.startswith("test_")]
    for test in tests:
        test()
    print(f"All {len(tests)} tests passed.")
