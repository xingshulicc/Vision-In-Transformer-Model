# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Thu Sep  1 09:47:53 2022

@author: DELL
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange

class OutlookAttention(nn.Module):
    def __init__(self, dim, num_heads, kernel_size, padding, stride, attn_drop, proj_drop):
        super().__init__()
        
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, 
                           padding=0, dilation=1, groups=1, bias=True)
        self.attn = nn.Conv2d(in_channels=dim, out_channels=kernel_size**4 * num_heads, kernel_size=1, 
                              stride=1, padding=0, dilation=1, groups=1, bias=False)
        
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, 
                              padding=0, dilation=1, groups=1, bias=False)
        self.proj_drop = nn.Dropout(p=proj_drop)
        
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=1, padding=padding, stride=stride)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        v = self.v(x)
        # the shape of v is: B, C, H, W
        v = self.unfold(v)
        # the shape of v is: B, C * kernel_size**2, L=new_H * new_W
        new_H = math.floor((H + 2 * self.padding - self.kernel_size + self.stride)/self.stride)
        new_W = math.floor((W + 2 * self.padding - self.kernel_size + self.stride)/self.stride)
        v = rearrange(v, 'b (c d) l -> b l d c', c=C)
        # the shape of v is: B, L, kernel_size**2, C
        v = rearrange(v, 'b l d (n h) -> b n l d h', n=self.num_heads)
        # the shape of v is: B, num_heads, L, kernel_size**2, head_dim
        
        x_pooled = F.adaptive_avg_pool2d(input=x, output_size=(new_H, new_W))
        attn = self.attn(x_pooled)
        # the shape of attn is: B, kernel_size**4 * num_heads, new_H, new_W
        attn = rearrange(attn, 'b (k n) h w -> b n (h w) k', n=self.num_heads)
        # the shape of attn is: B, num_heads, L=new_H * new_W, kernel_size**4
        attn = rearrange(attn, 'b n l (i d) -> b n l i d', d=self.kernel_size**2)
        # the shape of attn is: B, num_heads, L, kernel_size**2, kernel_size**2
        attn = F.softmax(input=attn, dim=-1)
        attn = self.attn_drop(attn)
        
        out = einsum('b n l i d, b n l d h -> b n l i h', attn, v)
        # the shape of out is: B, num_heads, L, kernel_size**2, head_dim
        out = rearrange(out, 'b n l i h -> b (n h) i l')
        # the shape of out is: B, C, kernel_size**2, L
        out = rearrange(out, 'b c i l -> b (c i) l')
        # the shape of out is: B, C * kernel_size**2, L
        out = F.fold(input=out, output_size=(H, W), kernel_size=self.kernel_size, 
                     dilation=1, padding=self.padding, stride=self.stride)
        # the shape of out is: B, C, H, W
        
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, 
                                stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.norm = nn.BatchNorm2d(num_features=hidden_channels)
        self.act = nn.GELU()
        self.conv_2 = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1, 
                                stride=1, padding=0, dilation=1, groups=1, bias=False)
        
    def forward(self, x):
        out = self.conv_1(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.conv_2(out)
        
        return out

class Outlooker(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads, 
                 kernel_size, 
                 padding, 
                 stride, 
                 attn_drop, 
                 proj_drop, 
                 mlp_hidden_ratio):
        super().__init__()
        
        self.norm_1 = nn.BatchNorm2d(num_features=dim)
        self.Attention = OutlookAttention(dim=dim, 
                                          num_heads=num_heads, 
                                          kernel_size=kernel_size, 
                                          padding=padding, 
                                          stride=stride, 
                                          attn_drop=attn_drop, 
                                          proj_drop=proj_drop)
        
        self.norm_2 = nn.BatchNorm2d(num_features=dim)
        self.mlp = MLP(in_channels=dim, 
                       hidden_channels=int(mlp_hidden_ratio * dim), 
                       out_channels=dim)
        
    def forward(self, x):
        out = self.norm_1(x)
        out = self.Attention(out)
        # the shape of out is: B, C, H, W
        out = out + x
        
        y = self.norm_2(out)
        y = self.mlp(y)
        y = y + out
        # the shape of y is: B, C, H, W
        
        return y

        



# if __name__  == "__main__":
#         modelviz = Outlooker(dim=32, 
#                              num_heads=8, 
#                              kernel_size=3, 
#                              padding=1, 
#                              stride=1, 
#                              attn_drop=0.25, 
#                              proj_drop=0.25, 
#                              mlp_hidden_ratio=3.0)
#         sampledata = torch.rand(2, 32, 224, 224)
#         out = modelviz(sampledata)
#         print(out)
        