import torch

from torch.utils.data import IterableDataset, DataLoader


from PIL import Image

import os



class Data(IterableDataset):

  def __init__(self, hf_dataset, processor, condition):

    self.dataset = hf_dataset

    self.processor = processor

    self.condition = condition

  def __iter__(self):
      worker_info = torch.utils.data.get_worker_info()

      if worker_info is None:
          # Single-process fallback (num_workers=0)
          current_dataset = self.dataset
      else:
          # Multi-process: Streams only the assigned web-shards to this specific worker thread!
          current_dataset = self.dataset.shard(
              num_shards=worker_info.num_workers, 
              index=worker_info.id
          )

      # Now you iterate through a clean, pre-split network stream without any modulo logic
      for item in current_dataset:
          if self.condition == "transformers":
            image = self.processor(images=item['image'].convert('RGB'), return_tensors="pt")
            image['pixel_values'] = image['pixel_values'].squeeze(0).half()
          elif self.condition == "timm":
            image = self.processor(image)
          label = item['label']
          
          yield image, label





def prep_data(dataset, processor, condition):

    val_batch_size = 1000

    data = Data(dataset, processor, condition)

    DL = DataLoader(

        data,

        val_batch_size,

        shuffle=False,

        pin_memory=True,

        num_workers = min(16,os.cpu_count())

    )

    return DL