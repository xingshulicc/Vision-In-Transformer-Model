# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Wed Sep 28 13:26:52 2022

@author: DELL
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat

def get_relative_distance(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)
    coords = torch.meshgrid(coords_h, coords_w)
    coords = torch.stack(coords, dim=0)
    coords = torch.reshape(coords, shape=(2, -1))
    relative_distance = torch.sum(coords, dim=0)
    # the shape of relative_distance is: (window_size**2,)
    
    return relative_distance

class QnA(nn.Module):
    def __init__(self, batch_size, in_dim, out_dim, num_heads, kernel_size, padding, stride, drop_ratio, m):
        super().__init__()
        
        self.stride = stride
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.m = m
        
        self.q = nn.Parameter(torch.randn(batch_size, out_dim, 1))
        self.k_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, 
                                stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.v_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, 
                                stride=1, padding=0, dilation=1, groups=1, bias=False)
        
        self.relative_indices = get_relative_distance(kernel_size)
        self.relative_indices = self.relative_indices.type(torch.long)
        self.pos_embedding = nn.Parameter(torch.randn(1, 2 * kernel_size -1))
        
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=1, padding=padding, stride=stride)
        
        self.attn_drop = nn.Dropout(p=drop_ratio)
        
        self.proj = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=1, 
                              stride=1, padding=0, dilation=1, groups=1, bias=False)
        
        self.linear = nn.Linear(in_features=m, out_features=1, bias=False)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        k = self.k_conv(x)
        v = self.v_conv(x)
        # the shape of k and v is: B, out_dim, H, W
        
        k = self.unfold(k)
        v = self.unfold(v)
        # the shape of k and v is: B, out_dim * kernel_size**2, L=new_H * new_W
        
        new_H = math.floor((H - 1 + self.stride)/self.stride)
        new_W = math.floor((W - 1 + self.stride)/self.stride)
        
        q = repeat(self.q, 'b c d -> b l c d', l=new_H*new_W)
        q = rearrange(q, 'b l (h n) d -> b l h d n', h=self.num_heads)
        # the shape of q is: B, L, num_heads, 1, head_dim
        
        k = rearrange(k, 'b (c k) l -> b l c k', c=self.out_dim)
        k = rearrange(k, 'b l (h n) k -> b l h n k', h=self.num_heads)
        # the shape of k is: B, L, num_heads, head_dim, kernel_size**2
        
        v = rearrange(v, 'b (c k) l -> b l c k', c=self.out_dim)
        v = rearrange(v, 'b l (h n) k -> b l h k n', h=self.num_heads)
        # the shape of v is: B, L, num_heads, kernel_size**2, head_dim
        
        attn = torch.matmul(q, k)
        # the shape of attn is: B, L, num_heads, 1, kernel_size**2
        
        attn += self.pos_embedding[:, self.relative_indices[:]]
        attn = F.softmax(input=attn, dim=-1)
        
        attn = repeat(attn, 'b l h d k -> b l h (repeat d) k', repeat=self.m)
        attn = rearrange(attn, 'b l h m k -> b l h k m')
        attn = self.linear(attn)
        # the shape of attn is: B, L, num_heads, kernel_size**2, 1
        attn = rearrange(attn, 'b l h k d -> b l h d k')
        attn = self.attn_drop(attn)
        
        out = torch.matmul(attn, v)
        # the shape of out is: B, L, num_heads, 1, head_dim
        out = rearrange(out, 'b l h d n -> b (h n) d l')
        out = rearrange(out, 'b c d l -> b (c d) l')
        
        out = F.fold(input=out, output_size=(new_H, new_W), kernel_size=1, 
                     dilation=1, padding=0, stride=1)
        
        out = self.proj(out)
        
        return out



# if __name__  == "__main__":
#         modelviz = QnA(batch_size=2, 
#                        in_dim=32, 
#                        out_dim=64, 
#                        num_heads=8, 
#                        kernel_size=3, 
#                        padding=1, 
#                        stride=2, 
#                        drop_ratio=0.25, 
#                        m=12)
#         sampledata = torch.rand(2, 32, 224, 224)
#         out = modelviz(sampledata)
#         print(out)
