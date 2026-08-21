"""Model loaders for the two Vision Transformers studied here.

APE model  : google/vit-base-patch16-224 (learned absolute position embeddings),
             loaded through HuggingFace transformers. source tag "transformers".
RoPE model : vit_base_patch16_rope_224.naver_in1k (rotary position embeddings),
             loaded through timm. source tag "timm".

Both are ViT-Base/16 at 224 resolution with 12 layers, trained on ImageNet-1k,
so their heads produce ImageNet-1k logits directly. That lets us read top-1
accuracy for the fragility score without any finetuning, and it lets us compare
the two position encoding schemes on an equal footing.
"""

import torch
import torch.nn as nn


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_ape(model_name="google/vit-base-patch16-224", device=None, half=False):
    """Load the learned absolute position embedding ViT through transformers."""
    from transformers import ViTImageProcessor, ViTForImageClassification

    device = device or get_device()
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name).to(device)
    model.eval()
    if half:
        model = model.half()
    return model, processor, "transformers"

def load_ape_timm(model_name="vit_base_patch16_224.dino", device=None, half=False, input_size=None):
    """Load the learned absolute position embedding ViT through timm."""
    import timm

    device = device or get_device()
    model = timm.create_model(model_name, pretrained=True, dynamic_img_size=True).to(device)
    model.eval()
    if half:
        model = model.half()
    data_config = timm.data.resolve_model_data_config(model)
    if input_size:
        data_config["input_size"] = (3, input_size, input_size)
    processor = timm.data.create_transform(**data_config, is_training=False)
    return model, processor, "timm"


def load_rope(model_name="vit_base_patch16_rope_224.naver_in1k", device=None, half=False, input_size=None):
    """Load the rotary position embedding ViT through timm."""
    import timm

    device = device or get_device()
    model = timm.create_model(model_name, pretrained=True, dynamic_img_size=True).to(device)
    model.eval()
    if half:
        model = model.half()
    data_config = timm.data.resolve_model_data_config(model)
    if input_size:
        data_config["input_size"] = (3, input_size, input_size)
    processor = timm.data.create_transform(**data_config, is_training=False)
    return model, processor, "timm"


def load_model(kind, device=None, half=False):
    """Dispatch on a short name so scripts can take 'ape' or 'rope' from argv."""
    kind = kind.lower()
    if kind in ("ape", "transformers", "google"):
        return load_ape(device=device, half=half)
    if kind in ("rope", "timm", "naver"):
        return load_rope(device=device, half=half)
    if kind in ("dinov1", "ape_timm", "ape_second_seed"):
        return load_ape_timm(device=device, half=half)
    if kind in {"dinov3", "rope_second_seed"}:
        return load_rope(model_name="timm/vit_base_patch16_dinov3.lvd1689m", device=device, half=half)
    raise ValueError(f"unknown model kind {kind!r}; use 'ape', 'rope', 'ape_second_seed', or 'rope_second_seed'")


def num_prefix_tokens(model, source):
    """How many non patch tokens sit at the front of the sequence (the class
    token for both of these models). These are dropped before computing SSDC so
    that the token grid is a clean square."""
    if source == "transformers":
        return 1
    return int(getattr(model, "num_prefix_tokens", 1))


# --------------------------------------------------------------------------- #
# Model topology helpers.
#
# These locate the transformer blocks and their attention and MLP submodules in
# a way that survives library version differences. In particular the transformers
# ViT was refactored: older versions expose the blocks at `vit.encoder.layer`
# with the MLP residual add hidden inside a `ViTOutput` submodule, while newer
# versions expose them at `vit.layers` with a standalone `block.mlp` whose output
# is added to the residual. timm keeps `model.blocks` with `block.attn` and
# `block.mlp`. Everything downstream (SSDC capture, effective rank, ablation) goes
# through these helpers so there is a single place to adapt.
# --------------------------------------------------------------------------- #


def get_vit_blocks(model, source):
    """Return the ModuleList of transformer blocks."""
    if source == "timm":
        return model.blocks
    if source != "transformers":
        raise ValueError(f"source must be 'transformers' or 'timm', got {source!r}")

    base = getattr(model, "vit", model)
    encoder = getattr(base, "encoder", None)
    if encoder is not None and hasattr(encoder, "layer"):
        return encoder.layer  # older transformers: vit.encoder.layer
    if hasattr(base, "layers"):
        return base.layers  # newer transformers: vit.layers

    # Generic fallback: the longest ModuleList whose items look like ViT layers.
    best = None
    for module in model.modules():
        if isinstance(module, nn.ModuleList) and len(module) and "Layer" in type(module[0]).__name__:
            if best is None or len(module) > len(best):
                best = module
    if best is not None:
        return best
    raise AttributeError("could not locate the transformer blocks on this model")


def get_block_attention(block, source):
    """Return the attention submodule of one block."""
    if source == "timm":
        return block.attn
    return block.attention  # transformers, both old and new layouts


def get_patch_embed_conv(model, source):
    """Return the Conv2d that projects image patches into tokens.

    Its output has shape [B, C, H, W] before the flatten into a token sequence,
    which is where the RPI permutation acts. Robust to the transformers and timm
    attribute names, with a first Conv2d fallback.
    """
    if source == "timm":
        patch = getattr(model, "patch_embed", None)
        if patch is not None and hasattr(patch, "proj"):
            return patch.proj
    else:
        base = getattr(model, "vit", model)
        emb = getattr(base, "embeddings", None)
        patch = getattr(emb, "patch_embeddings", None) if emb is not None else None
        if patch is not None and hasattr(patch, "projection"):
            return patch.projection

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            return module
    raise AttributeError("could not locate the patch embedding convolution")


def get_block_mlp(block, source):
    """Return (module, mode) describing how to ablate the MLP of one block.

    mode "zero"     : the module output is added to the residual, so ablating it
                      means returning zeros (timm, and newer transformers).
    mode "residual" : the module performs the residual add internally (older
                      transformers ViTOutput), so ablating it means returning its
                      residual input.
    """
    if source == "timm":
        return block.mlp, "zero"
    if hasattr(block, "mlp"):  # newer transformers: standalone MLP added after
        return block.mlp, "zero"
    if hasattr(block, "output"):  # older transformers: ViTOutput adds internally
        return block.output, "residual"
    raise AttributeError("could not locate the MLP submodule on this block")
