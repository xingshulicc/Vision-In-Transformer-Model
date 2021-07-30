# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Fri Jul 30 11:40:57 2021

@author: default
"""
import torch
from torch import nn
from Patch_Merge import PatchMerging
from Swin_Transformer import SwinBlock

class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super(StageModule, self).__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)
# the output shape of StageModule: (batch_size, channels, height, width)

'''
how to use this module:
    input hyper-parameters:
        
        in_channels: int, the number of input channels
        
        hidden_dimension: int, it is the output channels of PatchMerging, and also the input channels for SwinBlock
        
        layers: int, its value should be divisible by 2, layers // 2 = the number of W-MSA and SW-MSA blocks
        
        downscaling_factor: int, spatial dimension reduction ratio
        
        num_heads: int, the number of heads for multi-head self-attention
        
        head_dim: int, default is 32
        
        window_size: int default is 7
        
        relative_pos_embedding: Boolean, use relative position embedding or not
        
    note that: the hyper-parameter mlp_dim in SwinBlock is hidden_dimension * 4 (we can change the value of 4 to reduce computation costs)
        
    if the input shape of x is (batch_size, channels, height, width), 
        first, through the PatchMerging, the output shape becomes (batch_size, channels * downscaling_factor **2, height // downscaling_factor, width // downscaling_factor)
        -> permute -> Linear layer, then the output shape becomes (batch_size, height // downscaling_factor, width // downscaling_factor, hidden_dimension)
        
        then, through the SwinBlock, the output shape becomes (batch_size, height // downscaling_factor, width // downscaling_factor, hidden_dimension)
        -> permute -> (batch_size, hidden_dimension, height // downscaling_factor, width // downscaling_factor)
        
        the number of input channels for the second StageModule is hidden_dimension, the height and width
        has been reduced to the height // downscaling_factor and width // downscaling_factor, so we need to increase
        the value of hidden_dimension in the second StageModule to hidden_dimension * 2
        
'''
