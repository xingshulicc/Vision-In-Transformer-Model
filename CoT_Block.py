# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Wed Aug 11 11:17:52 2021

@author: default
"""
import torch
from torch import nn, einsum
from einops import rearrange

class CoT_Mixer(nn.Module):
    def __init__(self, dim, kernel_size, groups, window_size, num_head, reduction_ratio):
        super(CoT_Mixer, self).__init__()
        
        self.num_head = num_head
        hidden_dim = self.num_head * window_size ** 2
        
        self.key_embed = nn.Sequential(
            nn.Conv2d(in_channels = dim, 
                      out_channels = dim, 
                      kernel_size = kernel_size, 
                      stride = 1, 
                      padding = kernel_size // 2, 
                      dilation = 1, 
                      groups = groups, 
                      bias = False, ), 
            nn.BatchNorm2d(num_features = dim), 
            nn.ReLU(inplace = True)
            )
        
        self.attn_map = nn.Sequential(
            nn.Conv2d(in_channels = dim * 2, 
                      out_channels = dim, 
                      kernel_size = 1, 
                      stride = 1, 
                      padding = 0, 
                      dilation = 1, 
                      groups = 1, 
                      bias = False, ), 
            nn.BatchNorm2d(num_features = dim), 
            nn.ReLU(inplace = True), 
            nn.Conv2d(in_channels = dim, 
                      out_channels = hidden_dim, 
                      kernel_size = 1, 
                      stride = 1, 
                      padding = 0, 
                      dilation = 1, 
                      groups = 1, 
                      bias = False, ), 
            nn.BatchNorm2d(num_features = hidden_dim)
            )
        
        self.value_embed = nn.Sequential(
            nn.Conv2d(in_channels = dim, 
                      out_channels = dim, 
                      kernel_size = 1, 
                      stride = 1, 
                      padding = 0, 
                      dilation = 1, 
                      groups = 1, 
                      bias = False, ), 
            nn.BatchNorm2d(num_features = dim)
            )
        
        self.pool = nn.AdaptiveAvgPool2d(output_size = window_size)
        
        self.norm = nn.LayerNorm(normalized_shape = dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features = dim, 
                      out_features = dim // reduction_ratio, 
                      bias = False), 
            nn.GELU(), 
            nn.Linear(in_features = dim // reduction_ratio, 
                      out_features = dim, 
                      bias = False)
            )
        
    def forward(self, x):
        
        B, C, H, W = x.shape
        
        k = self.key_embed(x)
        attn_embed = torch.cat([x, k], dim=1)
        qk = self.attn_map(attn_embed)
        # the output shape of qk is: (batch_size, hidden_dim, height, width)
        qk = qk.flatten(2).transpose(1, 2)
        # the output shape of qk is: (batch_size, height * width, hidden_dim)
        # hidden_dim = num_head * window_size ** 2
        qk = rearrange(qk, 'b i (h d) -> b h i d', h=self.num_head)
        
        x_ = self.pool(x)
        v = self.value_embed(x_)
        # the output shape of v is: (batch_size, dim, window_size, window_size)
        v = v.flatten(2).transpose(1, 2)
        # the output shape of v is: (batch_size, window_size ** 2, dim)
        v = rearrange(v, 'b d (h j) -> b h d j', h=self.num_head)
        
        qkv = einsum('b h i d, b h d j -> b h i j', qk, v)
        qkv = rearrange(qkv, 'b h i j -> b i (h j)')
        qkv = self.norm(qkv).transpose(1, 2)
        # the output shape of qkv is: (batch_size, dim, height * width)
        
        out = self.mlp(qkv)
        out = self.norm(out.transpose(1, 2))
        
        out = rearrange(out, 'b (n_h n_w) c -> b c n_h n_w', n_h=H)
        # the output shape of out is: (batch_size, dim, height, width)
        out = x + out
        
        return out


