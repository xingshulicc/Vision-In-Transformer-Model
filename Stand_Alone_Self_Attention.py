# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Tue Sep 13 14:23:45 2022

@author: DELL
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange

def get_relative_distance(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)
    coords = torch.meshgrid(coords_h, coords_w)
    coords = torch.stack(coords, dim=0)
    coords = torch.reshape(coords, shape=(2, -1))
    relative_distance = torch.sum(coords, dim=0)
    # the shape of relative_distance is: (window_size**2,)
    
    return relative_distance
    
    
class SASA(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, kernel_size, padding, stride, drop_ratio):
        super().__init__()

        self.stride = stride
        self.num_heads = num_heads
        self.out_dim = out_dim
        
        self.q_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, 
                                stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.k_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, 
                                stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.v_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, 
                                stride=1, padding=0, dilation=1, groups=1, bias=False)
        
        self.relative_indices = get_relative_distance(kernel_size)
        self.relative_indices = self.relative_indices.type(torch.long)
        self.pos_embedding = nn.Parameter(torch.randn(1, 2 * kernel_size -1))
        
        self.q_unfold = nn.Unfold(kernel_size=1, dilation=1, padding=0, stride=stride)
        self.k_unfold = nn.Unfold(kernel_size=kernel_size, dilation=1, padding=padding, stride=stride)
        self.v_unfold = nn.Unfold(kernel_size=kernel_size, dilation=1, padding=padding, stride=stride)
        
        self.attn_drop = nn.Dropout(p=drop_ratio)
        
        self.proj = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=1, 
                              stride=1, padding=0, dilation=1, groups=1, bias=False)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)
        # the shape of q, k and v is: B, out_dim, H, W
        
        q = self.q_unfold(q)
        # the shape of q is: B, out_dim, L=new_H * new_W
        
        k = self.k_unfold(k)
        v = self.v_unfold(v)
        # the shape of k and v is: B, out_dim * kernel_size**2, L=new_H * new_W
        
        new_H = math.floor((H - 1 + self.stride)/self.stride)
        new_W = math.floor((W - 1 + self.stride)/self.stride)
        
        q = rearrange(q, 'b (h d) l -> b l h d', h=self.num_heads)
        q = rearrange(q, 'b l h d -> b l h 1 d')
        
        k = rearrange(k, 'b (c k) l -> b l c k', c=self.out_dim)
        k = rearrange(k, 'b l (h d) k -> b l h d k', h=self.num_heads)
        
        v = rearrange(v, 'b (c k) l -> b l c k', c=self.out_dim)
        v = rearrange(v, 'b l (h d) k -> b l h k d', h=self.num_heads)
        
        attn = torch.matmul(q, k)
        # the shape of attn is: B, L, num_heads, 1, kernel_size**2
        attn += self.pos_embedding[:, self.relative_indices[:]]
        attn = F.softmax(input=attn, dim=-1)
        attn = self.attn_drop(attn)
        
        out = torch.matmul(attn, v)
        # the shape of out is: B, L, num_heads, 1, head_dim
        out = rearrange(out, 'b l h 1 d -> b (h d) 1 l')
        out = rearrange(out, 'b c 1 l -> b c l')
        # the shape of out is: B, out_dim, L=new_H * new_W
        
        out = F.fold(input=out, output_size=(new_H, new_W), kernel_size=1, 
                     dilation=1, padding=0, stride=1)
        # the shape of out is: B, out_dim, new_H, new_W
        
        out = self.proj(out)
        
        return out
        
        
# if __name__  == "__main__":
#         modelviz = SASA(in_dim=32, 
#                         out_dim=64, 
#                         num_heads=8, 
#                         kernel_size=3, 
#                         padding=1, 
#                         stride=2, 
#                         drop_ratio=0.25)
#         sampledata = torch.rand(2, 32, 224, 224)
#         out = modelviz(sampledata)
#         print(out)
