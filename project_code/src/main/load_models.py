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


def load_rope(model_name="vit_base_patch16_rope_224.naver_in1k", device=None, half=False):
    """Load the rotary position embedding ViT through timm."""
    import timm

    device = device or get_device()
    model = timm.create_model(model_name, pretrained=True).to(device)
    model.eval()
    if half:
        model = model.half()
    data_config = timm.data.resolve_model_data_config(model)
    processor = timm.data.create_transform(**data_config, is_training=False)
    return model, processor, "timm"


def load_model(kind, device=None, half=False):
    """Dispatch on a short name so scripts can take 'ape' or 'rope' from argv."""
    kind = kind.lower()
    if kind in ("ape", "transformers", "google"):
        return load_ape(device=device, half=half)
    if kind in ("rope", "timm", "naver"):
        return load_rope(device=device, half=half)
    raise ValueError(f"unknown model kind {kind!r}; use 'ape' or 'rope'")


def num_prefix_tokens(model, source):
    """How many non patch tokens sit at the front of the sequence (the class
    token for both of these models). These are dropped before computing SSDC so
    that the token grid is a clean square."""
    if source == "transformers":
        return 1
    return int(getattr(model, "num_prefix_tokens", 1))
