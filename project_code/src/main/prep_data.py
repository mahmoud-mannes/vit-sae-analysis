import torch

from torch.utils.data import IterableDataset, DataLoader

import sys

import os

import numpy as np

from PIL import Image as PILImage

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from make_imagenet_c import gaussian_blur, pixelate, fog, contrast, gaussian_noise, d

def apply_corruption(pil_img, corruption_fn, severity):
    """
    Takes a PIL Image, applies the ImageNet-C math, 
    and returns a clean PIL Image for your ViT processor.
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

  def __init__(self, hf_dataset, processor, source, corruption_type = None, severity = 5):

    self.dataset = hf_dataset

    self.processor = processor

    self.source = source

    self.corruption_fn = d[corruption_type] if corruption_type else None

    self.severity = severity

  def __iter__(self):
      worker_info = torch.utils.data.get_worker_info()

      if worker_info is None:
          # Single-process fallback (num_workers=0)
          current_dataset = self.dataset
      else:
          # Streams only the assigned web-shards to this specific worker thread
          current_dataset = self.dataset.shard(
              num_shards=worker_info.num_workers, 
              index=worker_info.id
          )

      
      for item in current_dataset:
          if self.corruption_fn:
             
             item['image'] = apply_corruption(item['image'], self.corruption_fn, self.severity)

          if self.source == "transformers":
            
            image = self.processor(images=item['image'].convert('RGB'), return_tensors="pt")
            image['pixel_values'] = image['pixel_values'].squeeze(0).half()
          
          elif self.source == "timm":

            image = self.processor(item['image'].convert('RGB')).half()
          
          label = item['label']
          
          yield image, label





def prep_data(dataset, processor, condition, corruption_type = None, severity = 5):

    val_batch_size = 1000

    data = Data(dataset, processor, condition, corruption_type, severity)

    DL = DataLoader(

        data,

        val_batch_size,

        shuffle=False,

        pin_memory=True,

        num_workers = min(24,os.cpu_count())

    )

    return DL