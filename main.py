import os
import sys

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from tqdm import tqdm

import datasets
import losses
import models
import options


def main():
  args = options.parser.parse_args(sys.argv[1:])
  os.makedirs(args.result_path, exist_ok=True)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # make dataloader
  train_dataset = datasets.RAW2RGB(args.dataset_path)
  train_loader = DataLoader(train_dataset, 
                            batch_size=args.batch_size,
                            shuffle=args.shuffle, 
                            num_workers=args.num_workers)

  # load color-mean and color-std of dataset
  #color_mean_path = os.path.join(args.dataset_path, 'color-mean.txt')
  #with open(color_mean_path, 'r') as f:
  #  color_mean = list(map(float, f.read().split()))
  #color_std = [1., 1., 1., 1.]
  
  # make model
  #model = models.EDSR(args.num_resblock, 
  #                    args.in_channels, args.out_channels, args.num_channels, 
  #                    color_mean, color_std, args.res_scale, args.scale)
  #model = nn.DataParallel(model, [0, 1, 2, 3])
  model = models.EDSR1(args.num_resblock, args.num_channels)
  model = model.to(device)

  optimizer = torch.optim.Adam(model.parameters())
  
  layers = {
    "conv_1_1": 1.0,
    "conv_3_2": 1.0
  }
  loss = losses.Contextual_Loss(layers, max_1d_size=64).to(device)
  
  for epoch in range(1, args.num_epochs+1):
    # train
    with tqdm(total=len(train_loader), desc='[%04d/%04d] train'%(epoch, args.num_epochs),
              unit='batch', ncols=96, position=0, miniters=1) as t:
      for batch_idx, (train, test) in enumerate(train_loader):
        # to device
        train = train.to(device)
        test = test.to(device)
        
        # forward
        result = model(train)
        
        # calculate loss
        cost = loss(result, test)

        # backward
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        # update tqdm
        cost_val = cost.item()
        t.set_postfix_str('loss %.04f'%cost_val)
        t.update()
    
    # TODO: validation
    
    # save test images
    result_image = transforms.ToPILImage()(result[0].cpu())
    result_image.save(os.path.join(args.result_path, '%05d-result.png'%epoch))
    result_image.close()
    train_image = transforms.ToPILImage()(train[0].cpu())
    train_image.save(os.path.join(args.result_path, '%05d-train.png'%epoch))
    train_image.close()
    test_image = transforms.ToPILImage()(test[0].cpu())
    test_image.save(os.path.join(args.result_path, '%05d-test.png'%epoch))
    test_image.close()
    
if __name__ == "__main__":
  main()
