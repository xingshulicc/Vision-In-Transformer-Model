# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 13:40:02 2021

@author: default
"""
import torch
from torch import nn, einsum
from einops import rearrange

def pair(x):
    return (x, x) if not isinstance(x, tuple) else x

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def rel_to_abs(x):
    b, h, l, _, device, dtype = *x.shape, x.device, x.dtype
    dd = {'device': device, 'dtype': dtype}
    col_pad = torch.zeros((b, h, l, 1), **dd)
    x = torch.cat((x, col_pad), dim=3)
    flat_x = rearrange(x, 'b h l c -> b h (l c)')
    flat_pad = torch.zeros((b, h, l - 1), **dd)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
    final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
    final_x = final_x[:, :, :l, (l - 1):]
    return final_x
# the output shape of rel_to_abs: [batch_size, heads * height, width, width]

def relative_logits_1d(q, rel_k):
    b, heads, h, w, dim = q.shape
    logits = einsum('bhxyd,rd->bhxyr', q, rel_k)
    logits = rearrange(logits, 'b h x y r -> b (h x) y r')
    logits = rel_to_abs(logits)
    logits = logits.reshape(b, heads, h, w, w)
    logits = expand_dim(logits, dim=3, k=h)
    return logits
# the output shape of relative_logits_1d: [batch_size, heads, height, height, width, width]

# dim_head = channels // heads
class RelPosEmb(nn.Module):
    def __init__(self, fmap_size, dim_head):
        super().__init__()
        height, width = pair(fmap_size)
        scale = dim_head ** -0.5
        self.fmap_size = fmap_size
        self.rel_height = nn.Parameter(torch.randn(2 * height - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(2 * width - 1, dim_head) * scale)
        
    def forward(self, q):
        h, w = self.fmap_size
        q = rearrange(q, 'b h (x y) d -> b h x y d', x = h, y = w)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b h x i y j -> b h (x y) (i j)')
        
        q = rearrange(q, 'b h x y d -> b h y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b h x i y j -> b h (y x) (j i)')
        
        return rel_logits_w + rel_logits_h
# the output shape of RelPosEmb: [batch_size, heads, height * width, height * width]
    
class AbsPosEmb(nn.Module):
    def __init__(self, fmap_size, dim_head):
        super().__init__()
        height, width = pair(fmap_size)
        scale = dim_head ** -0.5
        self.height = nn.Parameter(torch.randn(height, dim_head) * scale)
        self.width = nn.Parameter(torch.randn(width, dim_head) * scale)
        
    def forward(self, q):
        emb = rearrange(self.height, 'h d -> h () d') + rearrange(self.width, 'w d -> () w d')
        emb = rearrange(emb, 'h w d -> (h w) d')
        logits = einsum('bhid,jd->bhij', q, emb)
        return logits
    
# the output shape of AbsPosEmb: [batch_size, heads, height * width, height * width]

class Attention(nn.Module):
    def __init__(self, *, 
                 dim, 
                 fmap_size, 
                 heads, 
                 dim_head, 
                 rel_pos_emb=True):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        
        rel_pos_class = RelPosEmb if rel_pos_emb else AbsPosEmb
        
        self.pos_emb = rel_pos_class(fmap_size, dim_head)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape
        q, k, v = self.to_qkv(fmap).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), (q, k, v))
        q *= self.scale
        
        sim = einsum('bhid,bhjd->bhij', q, k)
        sim += self.pos_emb(q)
        attn = self.softmax(sim)
        
        out = einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return out

class BottleBlock(nn.Module):
    def __init__(self, *, 
                 dim, 
                 fmap_size, 
                 dim_out, 
                 proj_factor, 
                 stride, 
                 heads, 
                 dim_head, 
                 rel_pos_emb, 
                 activation=nn.ReLU()):
        super().__init__()
        if stride != 1 or dim != dim_out:
            self.shortcut = nn.Sequential(nn.Conv2d(dim, dim_out, kernel_size=1, stride=stride, bias=False), 
                                          nn.BatchNorm2d(dim_out), 
                                          activation)
        else:
            self.shortcut = nn.Identity()
        
        attn_dim_in = dim_out // proj_factor
        attn_dim_out = heads * dim_head
        
        self.net = nn.Sequential(nn.Conv2d(dim, attn_dim_in, 1, bias=False), 
                                 nn.BatchNorm2d(attn_dim_in), 
                                 activation, 
                                 Attention(dim=attn_dim_in, 
                                           fmap_size=fmap_size, 
                                           heads=heads, 
                                           dim_head=dim_head, 
                                           rel_pos_emb=rel_pos_emb), 
                                 nn.AvgPool2d((2, 2), (2, 2)) if stride != 1 else nn.Identity(), 
                                 nn.BatchNorm2d(attn_dim_out), 
                                 activation, 
                                 nn.Conv2d(attn_dim_out, dim_out, 1, bias=False), 
                                 nn.BatchNorm2d(attn_dim_out))
        nn.init.zeros_(self.net[-1].weight)
        self.activation = activation
        
    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.net(x)
        x += shortcut
        return self.activation(x)
    

