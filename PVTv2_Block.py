# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Fri Jul 30 11:53:17 2021

@author: xingshuli
"""

from torch import nn, einsum

from timm.models.layers import trunc_normal_
import math

from einops import rearrange

class DWConv(nn.Module):
    def __init__(self, dim):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(
            in_channels = dim, 
            out_channels = dim, 
            kernel_size = 3, 
            stride = 1, 
            padding = 1, 
            dilation = 1, 
            groups = dim, 
            bias = True, 
            )
        
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        
        return x
# the output shape of x is: (batch_size, N, channels)

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size):
        super(OverlapPatchEmbed, self).__init__()
        
        # the kernel_size, stride, padding should satisfy:
            # 2 * padding - kernel_size + stride = 0

        kernel_size = patch_size
        stride = math.ceil(kernel_size / 2)
        padding = math.floor(stride / 2)
        
        self.proj = nn.Conv2d(
            in_channels = in_channels, 
            out_channels = out_channels, 
            kernel_size = kernel_size, 
            stride = stride, 
            padding = padding, 
            dilation=1, 
            groups=1, 
            bias=True, 
            )
        self.norm = nn.LayerNorm(out_channels)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        
        return x, H, W
    # the output shape of x is: (batch_size, spatial dimension(H x W), channels)

class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias, attn_drop_rate, proj_drop_rate, pool_output_size):
        super(Attention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(in_features = dim, out_features = dim, bias = qkv_bias)
        self.kv = nn.Linear(in_features = dim, out_features = 2 * dim, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(in_features = dim, out_features = dim, bias = qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        
        self.pool = nn.AdaptiveAvgPool2d(output_size = pool_output_size)
        self.sr = nn.Conv2d(
            in_channels = dim, 
            out_channels = dim, 
            kernel_size = 1, 
            stride = 1, 
            padding = 0,
            dilation = 1, 
            groups = 1, 
            bias = False, 
            )
        self.norm = nn.LayerNorm(dim)
        self.act_func = nn.GELU()
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        
        Q = self.q(x)
        Q = rearrange(Q, 'b n (h d) -> b h n d', h=self.num_heads)
        
        x_ = x.permute(0, 2, 1)
        x_ = rearrange(x_, 'b c (h w) -> b c h w', h=H)
        x_reduce = self.pool(x_)
        x_reduce = self.sr(x_reduce)
        x_reduce = rearrange(x_reduce, 'b c n_h n_w -> b (n_h n_w) c')
        x_reduce = self.norm(x_reduce)
        x_reduce = self.act_func(x_reduce)
        
        K, V = self.kv(x_reduce).chunk(2, dim=-1)
        K = rearrange(K, 'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(V, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # perform self-attention operation
        attn = einsum('b h i d, b h j d -> b h i j', Q, K) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        out = einsum('b h i j, b h j d -> b h i d', attn, V)
        out = rearrange(out, 'b h i d -> b i (h d)')
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out
    # the shape of out is (B, N, C)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop_rate):
        super(MLP, self).__init__()
        
        self.act_func = nn.GELU()
        
        self.fc1 = nn.Linear(in_features = in_features, 
                             out_features = hidden_features, 
                             bias = False)
        self.dwconv = DWConv(hidden_features)
        self.fc2 = nn.Linear(in_features = hidden_features, 
                             out_features = out_features, 
                             bias = False)
        
        self.drop = nn.Dropout(drop_rate)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act_func(x)
        x = self.dwconv(x, H, W)
        x = self.act_func(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias, attn_drop_rate, proj_drop_rate, pool_output_size, 
                 mlp_hidden_ratio, mlp_drop_rate):
        super(Block, self).__init__()
        self.norm = nn.LayerNorm(dim)
        
        self.attention = Attention(dim, 
                                   num_heads, 
                                   qkv_bias, 
                                   attn_drop_rate, 
                                   proj_drop_rate, 
                                   pool_output_size)
        
        mlp_hidden_features = int(dim * mlp_hidden_ratio)
        self.mlp = MLP(in_features = dim, 
                       hidden_features = mlp_hidden_features, 
                       out_features = dim, 
                       drop_rate = mlp_drop_rate)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x, H, W):
        x += self.attention(self.norm(x), H, W)
        x += self.mlp(self.norm(x), H, W)
        
        return x
# the shape of x is (B, N, C)

class StageModule(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, 
                 num_heads, qkv_bias, attn_drop_rate, proj_drop_rate, pool_output_size, 
                 mlp_hidden_ratio, mlp_drop_rate):
        super(StageModule, self).__init__()
        
        self.patch_embed = OverlapPatchEmbed(in_channels, out_channels, patch_size)
        self.MHSA = Block(out_channels, num_heads, qkv_bias, attn_drop_rate, proj_drop_rate, pool_output_size, 
                          mlp_hidden_ratio, mlp_drop_rate)
        
    def forward(self, x):
        x, H, W = self.patch_embed(x)
        x = self.MHSA(x, H, W)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H)
        return x
'''
the input shape for StageModule should be: (batch_size, in_channels, height, width)
the output shape of StageModule is: (batch_size, ((height/stride) x (width / stride)), out_channels)
'''


