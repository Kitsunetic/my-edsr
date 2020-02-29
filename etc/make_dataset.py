import os
import pickle
import random
import shutil

import imageio
import numpy as np
import rawpy
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

src_dataset_path = '/mnt/d/datasets/SRRAW-align.v3'
dst_dataset_path = '/mnt/d/datasets/SRRAW128'

black_lv = 512
white_lv = 16384

dataset_size = 128
patch_size = 1024


def main():
  src_path_list = []
  failure_paths = []
  
  src_dirs = os.listdir(src_dataset_path)
  with tqdm(total=len(src_dirs), desc='search files', unit='dir') as t:
    for src_dir in src_dirs:
      src_dir_path = os.path.join(src_dataset_path, src_dir)
      if not os.path.isdir(src_dir_path):
        continue
      
      src_files = sorted(os.listdir(src_dir_path))
      src_lr_files = filter(lambda f: f.endswith('.ARW'), src_files)
      src_lr_paths = list(map(lambda f: os.path.join(src_dir_path, f), src_lr_files))
      src_hr_paths = list(map(lambda f: os.path.splitext(f)[0] + '.JPG', src_lr_paths))
      for lr, hr in zip(src_lr_paths, src_hr_paths):
        src_path_list.append((lr, hr))
        """ # when dataset contains crashed images
        try:
          with rawpy.imread(lr):
            src_path_list.append((lr, hr))
        except rawpy._rawpy.LibRawNonFatalError:
          failure_paths.append((lr, hr))
        """
      # update tqdm
      t.update()
  
  os.makedirs(dst_dataset_path, exist_ok=True)
  
  #color_mean = [0., 0., 0., 0.]
  
  src_paths = random.sample(src_path_list, dataset_size)
  with tqdm(total=dataset_size, desc='copy files', unit='file') as t:
    for i, (lr_path, hr_path) in enumerate(src_paths, 1):
      with rawpy.imread(lr_path) as raw:
        lr = np.array(raw.raw_image_visible.copy(), dtype=np.float32)
        lr = (lr-black_lv) / (white_lv-black_lv)
        lr = lr[8:-8, 8:-8]
      
      hr = np.array(imageio.imread(hr_path), dtype=np.float32)
      hr = hr / 255.0
      
      # some lr images have larger height than 448px
      if lr.shape[0] > hr.shape[0]:
        lr = lr[224:-224, :]
      
      """
      # make patch
      h, w = lr.shape
      dh = random.randint(0, h-patch_size)
      dw = random.randint(0, w-patch_size)
      hr = hr[dh:dh+patch_size, dw:dw+patch_size, :]
      lr = lr[dh:dh+patch_size, dw:dw+patch_size]
      """
      """
      # make lr 4channel
      lr_ = np.zeros((lr.shape[0]//2, lr.shape[1]//2, 4), dtype=np.float32)
      lr_[:, :, 0] = lr[0::2, 0::2] # R
      lr_[:, :, 1] = lr[1::2, 0::2] # G1
      lr_[:, :, 2] = lr[1::2, 1::2] # B
      lr_[:, :, 3] = lr[0::2, 1::2] # G2
      lr = lr_
      
      # get color mean
      color_mean[0] += lr[:, :, 0].mean() / dataset_size
      color_mean[1] += lr[:, :, 1].mean() / dataset_size
      color_mean[2] += lr[:, :, 2].mean() / dataset_size
      color_mean[3] += lr[:, :, 3].mean() / dataset_size
      """
      
      dst_path = os.path.join(dst_dataset_path, '%05d.pkl'%i)
      with open(dst_path, 'wb') as f:
        pickle.dump({'lr': lr, 'hr': hr}, f)
      
      # update tqdm
      t.update()
  
  """
  # save color-mean
  color_mean_path = os.path.join(dst_dataset_path, 'color-mean.txt')
  with open(color_mean_path, 'w') as f:
    f.write(' '.join(map(str, color_mean)))
  """
  """
  # save failed files
  with open('failure.txt', 'w') as f:
    for lr, hr in failure_paths:
      f.write('%s ; %s\n'%(lr, hr))
  """
  
if __name__ == "__main__":
  main()
