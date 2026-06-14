import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as T
from PIL import Image
import os

class Data(Dataset):
  def __init__(self, hf_dataset):
    self.dataset = hf_dataset
  def __len__(self):
    return len(self.dataset)
  def __getitem__(self, index):
     item = self.dataset[index]
     image = Image.open(item['image'].convert('RGB'))
     label = item['label']
     return image,label


def prep_data(dataset):
    val_batch_size = 1000
    data = Data(dataset)
    DL = DataLoader(data,val_batch_size,shuffle=False,pin_memory=False, num_workers = 0)
    return DL