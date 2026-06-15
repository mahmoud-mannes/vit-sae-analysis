import torch

from torch.utils.data import IterableDataset, DataLoader

from datasets.distributed import split_dataset_by_node

from PIL import Image

import os



class Data(IterableDataset):

  def __init__(self, hf_dataset, processor):

    self.dataset = hf_dataset

    self.processor = processor



  def __iter__(self):

    for item in self.dataset:

        image = self.processor(images = item['image'].convert('RGB'), return_tensors = "pt")

        image['pixel_values'] = image['pixel_values'].squeeze(0)

        label = item['label']

        yield image,label





def prep_data(dataset, processor):

    val_batch_size = 1000

    data = Data(dataset,processor)

    DL = DataLoader(

        split_dataset_by_node(data, rank=0, world_size=1),

        data,

        val_batch_size,

        shuffle=False,

        pin_memory=False,

        num_workers = min(16,os.cpu_count())

    )

    return DL