import torch

from torch.utils.data import IterableDataset, DataLoader, get_worker_info

import sys

import os

import numpy as np

from PIL import Image as PILImage

import multiprocessing

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))


def get_corruption_registry():
    """Return the name to corruption function map.

    Prefer the full ImageNet-C suite from make_imagenet_c.py when its heavy
    dependencies (ImageMagick / wand / opencv) are importable. Otherwise fall
    back to the light, pure Pillow / numpy corruptions in
    interventions.corruptions, which cover the shifts these experiments use and
    match the ImageNet-C math for Gaussian blur. This keeps the pipeline runnable
    on a clean Colab without extra system packages.
    """
    registry = {}
    try:
        from make_imagenet_c import d as imagenet_c_d

        registry.update(imagenet_c_d)
    except Exception:
        pass
    try:
        from interventions.corruptions import CORRUPTIONS

        for name, fn in CORRUPTIONS.items():
            registry.setdefault(name, fn)  # ImageNet-C wins if both are present
    except Exception:
        pass
    return registry

def apply_corruption(pil_img, corruption_fn, severity):
    """
    Takes a PIL Image, applies the ImageNet-C math, 
    and returns a clean PIL Image.
    """
    # Force RGB
    pil_img = pil_img.convert('RGB')
    
    # Apply corruption math (some take PIL, some take array)
    try:
        corrupted_arr = corruption_fn(pil_img, severity=severity)
    except TypeError:
        # Fallback for functions that strictly require a NumPy array input
        corrupted_arr = corruption_fn(np.array(pil_img), severity=severity)
        
    # Convert back to uint8 if the math yielded floats
    if isinstance(corrupted_arr, np.ndarray):
        if corrupted_arr.max() <= 1.01: 
            corrupted_arr = corrupted_arr * 255
        corrupted_arr = np.clip(corrupted_arr, 0, 255).astype(np.uint8)
        return PILImage.fromarray(corrupted_arr)
        
    return corrupted_arr # If it returned a PIL Image already (like jpeg_compression)


class Data(IterableDataset):

  def __init__(self, hf_dataset, processor, source, corruption_type=None, severity=5, number_images=None, half = True):

    self.dataset = hf_dataset

    self.processor = processor

    self.source = source

    self.corruption_fn = get_corruption_registry()[corruption_type] if corruption_type else None

    self.severity = severity

    self.number_images = number_images

    self.global_count = multiprocessing.Value('i', 0)

    self.half = half

  def __iter__(self):
      worker_info = get_worker_info()

      if worker_info is None:
          # Single-process loading: iterate the whole stream.
          dataset_iter = iter(self.dataset)
      else:
          # Multi-worker: each worker must only see its own shard of the
          # stream, otherwise every worker re-iterates the full stream and
          # you get num_workers duplicates of every image.
          # HF streaming datasets expose this for exactly this purpose.
          worker_dataset = self.dataset.shard(
              num_shards=worker_info.num_workers,
              index=worker_info.id,
          )
          dataset_iter = iter(worker_dataset)

      for item in dataset_iter:
          with self.global_count.get_lock():
            if self.number_images is not None and self.global_count.value >= self.number_images:
                break
            self.global_count.value += 1

          img = item['image'].convert('RGB')

          if self.corruption_fn:
              img = apply_corruption(img, self.corruption_fn, self.severity)

          if self.source == "transformers":
              image = self.processor(images=img, return_tensors="pt")
              image['pixel_values'] = image['pixel_values'].squeeze(0)

          elif self.source == "timm":
              image = self.processor(img)

          else:
              raise ValueError(f"Unknown source: {self.source!r}, expected 'transformers' or 'timm'")

          label = item['label']
        
          if self.half:
            yield image.half(), label
          else:
            yield image, label


def prep_data(dataset, processor, source, corruption_type=None, severity=5, number_images=None, batch_size=1000, half = True):

    data = Data(dataset, processor, source, corruption_type, severity, number_images, half)

    num_workers = min(dataset.n_shards, os.cpu_count())

    DL = DataLoader(
        data,
        batch_size=batch_size,
        # shuffle is not supported for IterableDataset. If you want
        # shuffling, do it upstream: hf_dataset.shuffle(buffer_size=...)
        # before passing it in here.
        pin_memory=True,
        num_workers=num_workers,
    )

    return DL