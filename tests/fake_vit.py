"""Shared tiny ViT fixtures for CPU-only intervention tests."""

import torch
import torch.nn as nn


def approx(left, right, tolerance=1e-6):
    return abs(float(left) - float(right)) <= tolerance


class TimmBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Linear(channels, channels)

    def forward(self, tokens):
        tokens = tokens + self.attn(self.norm1(tokens))
        return tokens + self.mlp(self.norm2(tokens))


class FakePatchEmbed(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Conv2d(3, channels, kernel_size=1, bias=False)


class FakeImageTimm(nn.Module):
    def __init__(self, channels=8, layers=2, grid_size=3):
        super().__init__()
        self.patch_embed = FakePatchEmbed(channels)
        self.blocks = nn.ModuleList([TimmBlock(channels) for _ in range(layers)])
        self.cls_token = nn.Parameter(torch.randn(1, 1, channels))
        self.patch_pos = nn.Parameter(
            torch.randn(1, grid_size * grid_size, channels)
        )
        self.head = nn.Linear(channels, 2)
        self.num_prefix_tokens = 1

    def forward(self, images):
        patches = self.patch_embed.proj(images).flatten(2).transpose(1, 2)
        patches = patches + self.patch_pos
        cls_token = self.cls_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat([cls_token, patches], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(tokens[:, 0])


def make_image_batches(seed=0, grid_size=3, num_images=6):
    torch.manual_seed(seed)
    images = torch.randn(num_images, 3, grid_size, grid_size)
    labels = torch.zeros(num_images, dtype=torch.long)
    return [(images, labels)]


def count_forward_hooks(model):
    return sum(
        len(getattr(module, "_forward_hooks", {}))
        + len(getattr(module, "_forward_pre_hooks", {}))
        for module in model.modules()
    )
