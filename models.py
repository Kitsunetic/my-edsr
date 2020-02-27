import math
from typing import Dict, List

import torch.nn as nn


class MeanShift(nn.Module):
  def __init__(self, color_mean: List[float], sign=1):
    super(MeanShift, self).__init__()
    self.color_mean = list(map(lambda c: c*sign, color_mean))
    self.sign = sign

  def forward(self, x):
    for i in range(len(self.color_mean)):
      x[i, :, :] += self.color_mean[i]*self.sign
    return x
    
class ResBlock(nn.Module):
  def __init__(self, num_channels: int, kernel_size: int, bias=True, res_scale=1):
    super(ResBlock, self).__init__()

    self.body = nn.Sequential([
      nn.Conv2d(num_channels, num_channels, kernel_size, bias=bias, padding=kernel_size//2),
      nn.ReLU(True),
      nn.Conv2d(num_channels, num_channels, kernel_size, bias=bias, padding=kernel_size//2),
    ])
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
               in_channels: int, out_channels: int, 
               num_channels: int, color_mean: List[float], 
               res_scale: int, scale: int):
    super(EDSR, self).__init__()
    
    # mean shift
    self.sub_mean = MeanShift(color_mean, sign=-1) # input 4channel
    self.add_mean = MeanShift([
      color_mean[0],
      (color_mean[1]+color_mean[3])/2, 
      color_mean[2]
    ], sign=1) # output 3channel
    
    # head module
    self.head_module = nn.Sequential([
      nn.Conv2d(in_channels, num_channels, 3, padding=1) # 4channel raw -> 12channel
    ])
    
    # body module
    self.body_module = nn.Sequential([
      ResBlock(num_channels, 3, res_scale=res_scale) for _ in range(num_resblock)
    ])
    
    # tail module
    self.tail_module = nn.Sequential([
      UpSampler(scale, num_channels),
      nn.Conv2d(num_channels, out_channels)
    ])

  def forward(self, x):
    x = self.sub_mean(x)
    x = self.head_module(x)
    
    res = self.body(x)
    res += x
    
    x = self.tail(res)
    x = self.add_mean(x)
    
    return x
