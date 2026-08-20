import os
import sys

import numpy as np
import torch
import torch.nn as nn

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "project_code", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from experiments.attention_output_analysis import (
    capture_block_stream_ssdc,
    resolve_block_stage_topology,
)
from metrics.position_probe import (
    evaluate_coordinate_probe,
    fit_shared_ridge_probes,
    fit_shared_ridge_probes_torch,
    split_image_indices,
)


class IdentityAttention(nn.Module):
    def forward(self, x):
        return x


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.Identity()
        self.attn = IdentityAttention()
        self.norm2 = nn.Identity()
        self.mlp = nn.Identity()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ScaleAttention(nn.Module):
    def forward(self, x):
        return 2.0 * x


class ScaleMLP(nn.Module):
    def forward(self, x):
        return 3.0 * x


class StageSemanticBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.Identity()
        self.attn = ScaleAttention()
        self.norm2 = nn.Identity()
        self.mlp = ScaleMLP()

    def forward(self, x):
        attn_out = self.attn(self.norm1(x))
        post_attn = x + attn_out
        mlp_out = self.mlp(self.norm2(post_attn))
        return post_attn + mlp_out


class TinyPatchEmbed(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Conv2d(3, channels, kernel_size=1, bias=False)
        with torch.no_grad():
            self.proj.weight.zero_()


class TinyImageTimm(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = TinyPatchEmbed(2)
        self.blocks = nn.ModuleList([TinyBlock(), TinyBlock()])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 2))
        self.patch_pos = nn.Parameter(
            torch.tensor(
                [
                    [
                        [0.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 0.0],
                        [1.0, 1.0],
                    ]
                ],
                dtype=torch.float32,
            )
        )
        self.head = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.head.weight.zero_()
        self.num_prefix_tokens = 1

    def forward(self, images):
        patches = self.patch_embed.proj(images).flatten(2).transpose(1, 2)
        patches = patches + self.patch_pos
        cls = self.cls_token.expand(images.shape[0], -1, -1)
        x = torch.cat([cls, patches], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.head(x[:, 0])


def test_block_stream_capture_reports_all_ssdc_curves_and_deltas():
    model = TinyImageTimm().eval()
    images = torch.zeros(5, 3, 2, 2)
    labels = torch.zeros(5, dtype=torch.long)
    batches = [(images, labels)]

    metrics = capture_block_stream_ssdc(model, "timm", batches, perm=None, half=False)

    assert set(metrics) == {
        "block_input",
        "post_attention_residual",
        "block_output",
        "attention_output",
        "mlp_output",
        "attention_residual_delta",
        "mlp_residual_delta",
        "mlp_residual_delta_well_defined",
    }
    for key in (
        "block_input",
        "post_attention_residual",
        "block_output",
        "attention_output",
        "mlp_output",
    ):
        assert len(metrics[key]) == 2
        assert metrics[key][0] > 0.45
        assert metrics[key][1] > 0.45
    assert metrics["mlp_residual_delta_well_defined"] is True
    assert len(metrics["attention_residual_delta"]) == 2
    assert len(metrics["mlp_residual_delta"]) == 2


def test_stage_topology_captures_explicit_block_states():
    block = StageSemanticBlock().eval()
    topology = resolve_block_stage_topology(block, "timm")
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
    seen = {}

    handles = [
        block.register_forward_pre_hook(
            lambda module, inputs: seen.setdefault(
                "block_input", inputs[0].detach().clone()
            )
        ),
        topology["attention_module"].register_forward_hook(
            lambda module, inputs, output: seen.setdefault("attention_output", output.detach().clone())
        ),
        topology["post_attention_norm"].register_forward_pre_hook(
            lambda module, inputs: seen.setdefault("post_attention_residual", inputs[0].detach().clone())
        ),
        topology["mlp_output_module"].register_forward_hook(
            lambda module, inputs, output: seen.setdefault("mlp_output", output.detach().clone())
        ),
        block.register_forward_hook(
            lambda module, inputs, output: seen.setdefault("block_output", output.detach().clone())
        ),
    ]
    try:
        block(x)
    finally:
        for handle in handles:
            handle.remove()

    assert torch.allclose(seen["block_input"], x)
    assert torch.allclose(seen["attention_output"], 2.0 * x)
    assert torch.allclose(seen["post_attention_residual"], 3.0 * x)
    assert torch.allclose(seen["mlp_output"], 9.0 * x)
    assert torch.allclose(seen["block_output"], 12.0 * x)


def test_split_image_indices_is_disjoint_and_image_level():
    split = split_image_indices(10, seed=7)

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


def test_probe_fit_beats_deterministic_shuffled_label_control():
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
        train_outputs, eval_outputs, split, shuffle_seed=17
    )
    metrics_repeat = evaluate_coordinate_probe(
        train_outputs, eval_outputs, split, shuffle_seed=17
    )

    assert metrics["row"]["r2"] > 0.98
    assert metrics["column"]["r2"] > 0.98
    assert metrics["negative_control"]["row"]["r2"] < 0.1
    assert metrics["negative_control"]["column"]["r2"] < 0.1
    assert metrics["negative_control"] == metrics_repeat["negative_control"]


def test_torch_ridge_matches_numpy_ridge_on_cpu():
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
    torch_result = fit_shared_ridge_probes_torch(
        train_x,
        val_x,
        test_x,
        targets,
        alpha_grid=(1e-2, 1e-1, 1.0),
        device="cpu",
    )

    assert torch_result["position"]["alpha"] == numpy_result["position"]["alpha"]
    assert abs(torch_result["position"]["r2"] - numpy_result["position"]["r2"]) < 1e-4

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


if __name__ == "__main__":
    tests = [value for key, value in sorted(globals().items()) if key.startswith("test_")]
    for test in tests:
        test()
    print(f"All {len(tests)} tests passed.")
