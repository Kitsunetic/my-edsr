import os
import pickle
import re

import torch


class RAW2RGB(torch.utils.data.Dataset):
  def __init__(self, dataset_path: str, black_lv=512, white_lv=16384):
    self.black_lv = black_lv
    self.white_lv = white_lv
    
    self.file_list = []
    for file in os.listdir(dataset_path):
      if re.match('\\d{5}\\.pkl', file):
        self.file_list.append(os.path.join(dataset_path, file))

  def __getitem__(self, idx: int):
    with open(self.file_list[idx], 'rb') as f:
      data = pickle.load(f)
    
    raw = data['input']
    target = data['target']
    
    return raw, target

  def __len__(self):
    return len(self.file_list)
