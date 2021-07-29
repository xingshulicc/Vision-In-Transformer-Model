# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Fri Jul  9 16:30:17 2021

@author: default
"""

import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat

# the shape of input x: [batch_size, height, width, channels]
class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super(CyclicShift, self).__init__()
        self.displacement = displacement
    
    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))

# the fn is a function
class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# the fn is a function, dim is input_features
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# transformer FFN part
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, False), 
            nn.GELU(), 
            nn.Linear(hidden_dim, dim, False), 
            )
        
    def forward(self, x):
        return self.net(x)

def create_mask(window_size, displacement, upper_lower, left_right):
    '''
    The implementation detail can refer to:
    https://zhuanlan.zhihu.com/p/361366090
    Parameters
    ----------
    window_size : int
        the size of local window for self-attention.
    displacement : int
        the value of top-left shift.
    upper_lower : Boolean
    left_right : Boolean

    '''
    mask = torch.zeros(window_size ** 2, window_size ** 2)
    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')
    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')
        
    return mask

def get_relative_distance(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    # the shape of indices: [window_size * window_size, 2]
    distances = indices[None, :, :] - indices[:, None, :]
    # the shape of indices[None, :, :]: [1, window_size * window_size, 2]
    # the shape of indices[:, None, :]: [window_size * window_size, 1, 2]
    # the shape of distances: [window_size * window_size, window_size * window_size, 2]
    
    # the value range of distances is: [-window_size + 1, window_size -1]
    return distances

class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super(WindowAttention, self).__init__()
        
        inner_dim = heads * head_dim
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.shifted = shifted
        self.relative_pos_embedding = relative_pos_embedding
        
        if self.shifted:
            displacement = window_size // 2
            # step_1: cyclic shift operation
            self.cyclic_shift = CyclicShift(-displacement)
            self.reverse_cyclic_shift = CyclicShift(displacement)
            
            # step_2: mask operation
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, 
                                                             displacement=displacement, 
                                                             upper_lower=True, 
                                                             left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(
                window_size=window_size, 
                displacement=displacement, 
                upper_lower=False, 
                left_right=True
                ), requires_grad=False)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        # the shape of pos_embedding is: (window_size ** 2, window_size ** 2)
        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distance(window_size) + window_size -1
            self.relative_indices = self.relative_indices.type(torch.long)
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size -1, 2 * window_size -1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))
        
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
    
        # the input shape of x should be: (batch_size, height, width, channels)-> channel_last order
    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)
        
        b, n_h, n_w, _, h = *x.shape, self.heads
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size
        
        # perform local window self-attention
        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d', 
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)
        # the shape of q, k, v is: (batch_size, heads, num_local_windows, local_window_dim, inner_dim)
        # local_window_dim is: (window_size ** 2, window_size ** 2)
        
        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale
        # the dots is calculated by: q(transpose(k))
        # the shape of dots is: (batch_size, heads, num_local_windows, local_window_dim, local_window_dim)
        
        # add position embedding
        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding
        
        # if implement window shift, then we need mask to remove cross-window attention
        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask
        
        # generate self-attention map
        attn = dots.softmax(dim=-1)
        # generate self-attention values
        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)', 
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        # generate final out, use linear operation to perform cross heads fusion
        out = self.to_out(out)
        
        # perform reverse window shift operation
        if self.shifted:
            out = self.reverse_cyclic_shift(out)
        
        return out

class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super(SwinBlock, self).__init__()
        
        self.attention_block = Residual(PreNorm(dim, WindowAttention(
            dim = dim, 
            heads=heads, 
            head_dim=head_dim, 
            shifted=shifted, 
            window_size=window_size, 
            relative_pos_embedding=relative_pos_embedding
            )))
        
        self.mlp_block = Residual(PreNorm(dim, FeedForward(
            dim=dim, 
            hidden_dim=mlp_dim
            )))
        
    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        
        return x



