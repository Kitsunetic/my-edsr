import math
from typing import Dict, List

import torch
import torch.nn as nn


class MeanShift(nn.Conv2d):
  def __init__(self, num_channels: int, color_mean: torch.Tensor, color_std: torch.Tensor, sign=1):
    super(MeanShift, self).__init__(num_channels, num_channels, 1)
    
    color_mean = torch.Tensor(color_mean)
    color_std = torch.Tensor(color_std)
    
    self.weight.data = torch.eye(num_channels).view(num_channels, num_channels, 1, 1) / color_std.view(num_channels, 1, 1, 1)
    self.bias.data = sign * 255 * color_mean / color_std
    for p in self.parameters():
      p.requires_grad = False

class ResBlock(nn.Module):
  def __init__(self, num_channels: int, kernel_size: int, bias=True, res_scale=1):
    super(ResBlock, self).__init__()

    self.body = nn.Sequential(
      nn.Conv2d(num_channels, num_channels, kernel_size, bias=bias, padding=kernel_size//2),
      nn.ReLU(True),
      nn.Conv2d(num_channels, num_channels, kernel_size, bias=bias, padding=kernel_size//2),
    )
    self.res_scale = res_scale

  def forward(self, x):
    res = self.body(x).mul(self.res_scale)
    res += x
    return res

class UpSampler(nn.Sequential):
  def __init__(self, scale: int, num_channels: int, bias=True):
    m = []
    if (scale & (scale-1)) == 0: # is scale = 2^n?
      for _ in range(int(math.log(scale, 2))):
        m.append(nn.Conv2d(num_channels, 4*num_channels, 3, padding=1, bias=bias))
        m.append(nn.PixelShuffle(2))
        m.append(nn.PReLU(num_channels))
    elif scale == 3:
      m.append(nn.Conv2d(num_channels, 9*num_channels, 3, padding=1, bias=bias))
      m.append(nn.PixelShuffle(3))
      m.append(nn.PReLU(num_channels))
    else:
      raise NotImplementedError('scale must 2^n or 3 integer.')
    
    super(UpSampler, self).__init__(*m)

class EDSR(nn.Module):
  def __init__(self, num_resblock: int, 
               in_channels: int, out_channels: int, num_channels: int, 
               color_mean: torch.Tensor, color_std: torch.Tensor, res_scale: int, scale: int):
    super(EDSR, self).__init__()
    
    # mean shift
    self.sub_mean = MeanShift(in_channels, color_mean, color_std, sign=-1) # input 4channels
    self.head_module = nn.Sequential(
      nn.Conv2d(in_channels, num_channels, 3, padding=1)
    )
    self.body_module = nn.Sequential(
      *[ResBlock(num_channels, 3, res_scale=res_scale) for _ in range(num_resblock)]
    )
    self.tail_module = nn.Sequential(
      UpSampler(scale, num_channels),
      nn.Conv2d(num_channels, out_channels, 3, padding=1)
    )
    self.add_mean = MeanShift(out_channels, 
                              [color_mean[0], (color_mean[1]+color_mean[3])/2, color_mean[2]], 
                              [color_std[0],  (color_std[1]+color_std[3])/2,   color_std[2] ], 
                              sign=1) # output 3channels

  def forward(self, x):
    x = self.sub_mean(x)
    x = self.head_module(x)
    
    res = self.body_module(x)
    res += x
    
    x = self.tail_module(res)
    x = self.add_mean(x)
    
    return x
