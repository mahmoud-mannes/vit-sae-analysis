"""Light, dependency slim input corruptions for the robustness measurements.

The original make_imagenet_c.py pulls in ImageMagick / wand / opencv, which is
awkward to install on a clean Colab. This module reimplements only the few
corruptions we actually use, in pure numpy / Pillow / scikit-image. The Gaussian
blur reproduces the ImageNet-C math (Hendrycks and Dietterich, 2019) exactly, so
fragility numbers line up with the reference runs, and prep_data falls back to
this module when the full ImageNet-C suite is not importable.

Every function takes a PIL RGB image plus a severity in 1..5 and returns a PIL
RGB image.
"""

from io import BytesIO

import numpy as np
from PIL import Image

try:
    from skimage.filters import gaussian as _sk_gaussian

    _HAVE_SKIMAGE = True
except Exception:  # pragma: no cover - depends on the environment
    from scipy.ndimage import gaussian_filter as _scipy_gaussian

    _HAVE_SKIMAGE = False


def _to_pil(arr):
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def gaussian_blur(img, severity=5):
    """ImageNet-C Gaussian blur. severity 1..5 maps to sigma in [1, 2, 3, 4, 6]."""
    sigma = [1, 2, 3, 4, 6][severity - 1]
    x = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    if _HAVE_SKIMAGE:
        x = _sk_gaussian(x, sigma=sigma, channel_axis=-1)
    else:
        x = np.stack([_scipy_gaussian(x[..., c], sigma=sigma) for c in range(3)], axis=-1)
    return _to_pil(x * 255.0)


def jpeg_compression(img, severity=5):
    """ImageNet-C JPEG. severity 1..5 maps to quality in [25, 18, 15, 10, 7]."""
    quality = [25, 18, 15, 10, 7][severity - 1]
    buf = BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def pixelate(img, severity=5):
    """ImageNet-C pixelate. severity 1..5 maps to a downscale factor."""
    factor = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    img = img.convert("RGB")
    w, h = img.size
    small = img.resize((max(1, int(w * factor)), max(1, int(h * factor))), Image.BOX)
    return small.resize((w, h), Image.BOX)


def patch_shuffle(img, severity=5, grid=None, seed=None):
    """Shuffle image patches on a coarse grid.

    This is a content disrupting shift analogous to RPI but applied to pixels: it
    scrambles where local content sits while leaving the global palette intact,
    so it probes reliance on local content vs global layout at the input. It is
    an extension beyond the reference corruptions. severity picks the grid size.
    """
    grid = grid if grid is not None else [2, 3, 4, 7, 14][severity - 1]
    img = img.convert("RGB")
    w, h = img.size
    cw, ch = (w // grid) * grid, (h // grid) * grid
    arr = np.array(img.resize((cw, ch), Image.BILINEAR))
    pw, ph = cw // grid, ch // grid
    blocks = [
        arr[r * ph : (r + 1) * ph, c * pw : (c + 1) * pw]
        for r in range(grid)
        for c in range(grid)
    ]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(blocks))
    out = np.zeros_like(arr)
    for dst, src in enumerate(order):
        r, c = divmod(dst, grid)
        out[r * ph : (r + 1) * ph, c * pw : (c + 1) * pw] = blocks[src]
    return _to_pil(out).resize((w, h), Image.BILINEAR)


# Keys match the ImageNet-C naming used in make_imagenet_c.py so the two
# registries are interchangeable.
CORRUPTIONS = {
    "Gaussian Blur": gaussian_blur,
    "JPEG": jpeg_compression,
    "Pixelate": pixelate,
    "Patch Shuffle": patch_shuffle,
}
