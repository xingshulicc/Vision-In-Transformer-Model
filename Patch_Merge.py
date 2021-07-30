# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Thu Jul 29 18:21:28 2021

@author: default
"""
import torch
from torch import nn

class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super(PatchMerging, self).__init__()
        
        self.downscaling_factor = downscaling_factor
        
        self.patch_merge = nn.Unfold(kernel_size = downscaling_factor, 
                                     dilation=1, 
                                     padding=0, 
                                     stride = downscaling_factor)
        # the shape of patch_merge is: (batch_size, channels * downscaling_factor ** 2, (input_dim / downscaling_factor) ** 2)
        # input_dim = height * width
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)
        
    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        # the shape of x is: (b, c*(downscaling_factor ** 2), new_h, new_w) -> permute to channels last order
        x = self.linear(x)
        # the shape of x is: (b, new_h, new_w, out_channels)
        
        return x

