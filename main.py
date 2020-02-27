import os
import sys

import imageio
import numpy as np
import torch
import torch.nn as nn
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
  dataloader = torch.utils.data.DataLoader(dataset, 
                                           batch_size=args.batch_size, 
                                           shuffle=args.shuffle, 
                                           num_workers=args.num_workers)

  # load color-mean of dataset
  color_mean_path = os.path.join(args.dataset_path, 'color_mean.txt')
  with open(color_mean_path, 'r') as f:
    color_mean = list(map(float, f.read().split()))

  # make model
  model = models.EDSR(args.num_resblock, 
                      args.in_channels, args.out_channels, args.num_channels, 
                      color_mean, args.res_scale, args.scale).to(device)

  optimizer = torch.optim.Adam(model.parameters())
  
  layers = {
    "conv_1_1": 1.0,
    "conv_3_2": 1.0
  }
  loss = losses.Contextual_Loss(layers, max_1d_size=64).to(device)
  
  history = {'cost': []}
  for epoch in range(1, args.num_epochs+1):
    with tqdm(total=len(dataloader), desc='[%04d/%04d]'%(epoch, args.num_epochs),
              unit='step', ncols=128, position=0, miniters=1) as t:
      for step, (train, test) in enumerate(dataloader):
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
        t.set_postfix_str('cost %.04f'%cost_val)
        t.update()
    
      # save result
      if epoch % 1 == 0:
        test_image = transforms.ToPILImage()(test[0].cpu())
        test_image.save('./results/result/test_%d.png'%epoch)
        test_image.close()
        
        result_image = transforms.ToPILImage()(result[0].cpu())
        result_image.save('./results/result/result_%d.png'%epoch)
        result_image.close()
      
    # TODO: make validation, save result as full image(not a patch size result)
    
if __name__ == "__main__":
  main()
