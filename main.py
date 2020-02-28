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
  dataset = datasets.RAW2RGB(args.dataset_path)
  
  valid_dataset_size = int(args.validation_split * len(dataset))
  train_dataset_size = len(dataset) - valid_dataset_size
  
  train_dataset, valid_dataset = random_split(dataset, 
                                              [train_dataset_size, valid_dataset_size])
  
  train_loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=args.shuffle, num_workers=args.num_workers)
  valid_loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=args.shuffle, num_workers=args.num_workers)

  # load color-mean and color-std of dataset
  color_mean_path = os.path.join(args.dataset_path, 'color-mean.txt')
  with open(color_mean_path, 'r') as f:
    color_mean = list(map(float, f.read().split()))
  color_std = [1., 1., 1., 1.]
  
  # make model
  model = models.EDSR(args.num_resblock, 
                      args.in_channels, args.out_channels, args.num_channels, 
                      color_mean, color_std, args.res_scale, args.scale).to(device)

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
    
    # validation
    cost_list = []
    psnr_list = []
    with tqdm(total=len(valid_loader), desc='[%04d/%04d] valid'%(epoch, args.num_epochs),
              unit='batch', ncols=96, position=0, miniters=1) as t:
      for batch_idx, (train, test) in enumerate(valid_loader):
        torch.set_grad_enabled(False)
        
        train = train.to(device)
        test = test.to(device)
        
        result = model(train)
        
        # calculate loss
        cost = loss(result, test)
        cost_list.append(cost.item())
        
        # calculate psnr
        psnr = (result - test) / 255.
        psnr = -10 * psnr.pow(2).mean().log10()
        psnr_list.append(psnr.item())
        
        # save results
        test_image = transforms.ToPILImage()(test[0].cpu())
        result_image = transforms.ToPILImage()(result[0].cpu())
        
        plt.figure(figsize=(16, 16))
        plt.subplot(1, 2, 1)
        plt.imshow(test_image)
        plt.subplot(1, 2, 2)
        plt.imshow(result_image)
        figure_path = os.path.join(args.result_path, 'epoch%04d-%04d.png'%(epoch, batch_idx))
        plt.savefig(figure_path, dpi=300)
        
        # update tqdm
        t.set_postfix_str('%04d-%04d'%(epoch, batch_idx))
        t.update()
        
        torch.set_grad_enabled(True)
      
      mean_loss = sum(cost_list) / len(cost_list)
      mean_psnr = sum(psnr_list) / len(psnr_list)
      t.set_postfix_str('val-loss %.4f psnr %.4f'%(mean_loss, mean_psnr))
    
if __name__ == "__main__":
  main()
