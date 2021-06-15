# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
from einops import rearrange

def rel_to_abs(x):
    '''
    convert relative indexing to absolute
    Input shape: [batch_size, heads * height, width, 2 * width -1]
    Ouput shape: [batch_size, heads * height, width, width]

    Denote: in BotNet paper, heads = heads * height
    relative indexing length = 2 * length -1, length = width
    
    More information about rel_to_abs can refer to: My github: xingshulicc

    '''
    bs, heads, length, _ = x.shape
    col_pad = tf.zeros((bs, heads, length, 1), dtype=x.dtype)
    x = tf.concat([x, col_pad], axis=3)
    # The shape of x: [bs, heads * height, width, 2 * width]
    flat_x = tf.reshape(x, [bs, heads, -1])
    # The shape of flat_x: [bs, heads * height, width * 2 * width]
    flat_pad = tf.zeros((bs, heads, length -1), dtype=x.dtype)
    flat_x_padded = tf.concat([flat_x, flat_pad], axis=2)
    # The shape of flat_x_padded: [bs, heads * height, width * 2 * width + width -1]
    final_x = tf.reshape(flat_x_padded, [bs, heads, length+1, 2*length-1])
    final_x = final_x[:, :, :length, length-1:]
    
    return final_x

def relative_logits_1d(*, q, rel_k, transpose_mask):
    '''
    compute relative logits along one dimension
    the shape of q: [batch_size, heads, height, width, dim]
    the shape of rel_k: [2 * width -1, dim]
    dim = num_channels // heads
    the output_shape: [batch_size, heads, height, width, height, width]
    transpose_mask: [0, 1, 2, 4, 3, 5]
    
    '''
    bs, heads, h, w, dim = q.shape
    rel_logits = tf.einsum('bhxyd,md->bhxym', q, rel_k)
    # The shape of rel_logits: [batch_size, heads, height, width, 2 * width -1]
    rel_logits = tf.reshape(rel_logits, [-1, heads * h, w, 2*w-1])
    # The shape of rel_logits: [batch_size, heads * height, width, 2 * width -1]
    rel_logits = rel_to_abs(rel_logits)
    # The shape of rel_logits: [batch_size, heads * height, width, width]
    rel_logits = tf.reshape(rel_logits, [-1, heads, h, w, w])
    rel_logits = tf.expand_dims(rel_logits, axis=3)
    # The shape of rel_logits: [batch_size, heads, height, 1, width, width]
    rel_logits = tf.tile(rel_logits, [1, 1, 1, h, 1, 1])
    # The shape of rel_logits: [batch_size, heads, height, height, width, width]
    rel_logits = tf.transpose(rel_logits, transpose_mask)
    
    return rel_logits

def relative_logits(q):
    '''
    Compute relative position enc logits.
    the shape of q: [batch_size, heads, height, width, dim]
    the output shape: [batch_size, heads, height, width, height, width]

    '''
    with tf.variable_scope('relative', reuse=tf.AUTO_REUSE):
        bs, heads, h, w, dim = q.shape
        int_dim = dim.value
        rel_emb_w = tf.get_variable('r_width', 
                                    shape=(2*w - 1, dim), 
                                    dtype=q.dtype, 
                                    initializer=tf.random_normal_initializer(stddev=int_dim**-0.5))
        rel_logits_w = relative_logits_1d(q=q, 
                                          rel_k=rel_emb_w, 
                                          transpose_mask=[0, 1, 2, 4, 3, 5])
        
        rel_emb_h = tf.get_variable('r_height', 
                                    shape=(2*h - 1, dim), 
                                    dtype=q.dtype, 
                                    initializer=tf.random_normal_initializer(stddev=int_dim**-0.5))
        rel_logits_h = relative_logits_1d(q=tf.transpose(q, [0, 1, 3, 2, 4]), 
                                          rel_k=rel_emb_h, 
                                          transpose_mask=[0, 1, 4, 2, 5, 3])
        
        return rel_logits_h + rel_logits_w

def relpos_self_attention(*, q, k, v, relative=True, fold_heads=True):
    '''
    2D self-attention with rel-pos
    the shape of q, k, v: [batch_size, heads, height, width, dim]
    the output shape: [batch_size, height, width, channels]
    channels = heads * dim

    '''
    bs, heads, h, w, dim = q.shape
    q = q * (dim ** -0.5) 
    logits = tf.einsum('bhHWd,bhPQd->bhHWPQ', q, k)
    
    if relative:
        logits += relative_logits(q)
    
    weights = tf.reshape(logits, [-1, heads, h, w, h * w])
    weights = tf.nn.softmax(weights)
    weights = tf.reshape(weights, [-1, heads, h, w, h, w])
    
    attn_out = tf.einsum('bhHWPQ,bhPQd->bHWhd', weights, v)
    
    if fold_heads:
        attn_out = tf.reshape(attn_out, [-1, h, w, heads * dim])
    
    return attn_out

def group_pointwise(featuremap, 
                    proj_factor=1, 
                    name='grouppoint', 
                    heads=4, 
                    target_dimension=None):
    '''
    1x1 conv with heads
    the input shape: [batch_size, height, width, channels]
    proj_factor is used to reduce dimension
    target_dimension: the output channels if defined in advance
    the output shape: [batch_size, heads, height, width, dim]
    output_channels = heads * dim
    
    this function is equal to ViT proj_dim + split_heads
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        in_channels = featuremap.shape[-1]
        
        if target_dimension is not None:
            proj_channels = target_dimension // proj_factor
        else:
            proj_channels = in_channels // proj_factor
        
        w = tf.get_variable('w', 
                            [in_channels, heads, proj_channels // heads], 
                            dtype=featuremap.dtype, 
                            initializer=tf.random_normal_initializer(stddev=0.01))
        
        out = tf.einsum('bHWD,Dhd->bhHWd', featuremap, w)
        return out

def absolute_logits(q):
    '''
    compute absolute position enc logits
    the shape of q: [batch_size, heads, height, width, dim]
    the output shape: [batch_size, heads, height, width, height, width]

    '''
    with tf.variable_scope('absolute', reuse=tf.AUTO_REUSE):
        bs, heads, h, w, dim = q.shape
        int_dim = dim.value
        
        emb_w = tf.get_variable('abs_width', 
                                shape=(w, dim), 
                                dtype=q.dtype, 
                                initializer=tf.random_normal_initializer(stddev=int_dim**-0.5))
        
        emb_h = tf.get_variable('abs_height', 
                                shape=(h, dim), 
                                dtype=q.dtype, 
                                initializer=tf.random_normal_initializer(stddev=int_dim**-0.5))
        
        emb_w = rearrange(emb_w, 'w d -> () w d')
        emb_h = rearrange(emb_h, 'h d -> h () d')
        # The shape of emb_w: [1, width, dim]
        # The shape of emb_h: [height, 1, dim]
        
        emb = emb_w + emb_h
        # The shape of emb: [height, width, dim]
        
        abs_logits = tf.einsum('bhxyd,pqd->bhxypq', q, emb)
        return abs_logits

def abspos_self_attention(*, q, k, v, absolute=True, fold_heads=True):
    '''
    2D self-attention with abs-pos
    the shape of q, k, v: [batch_size, heads, height, width, dim]
    the output shape: [batch_size, height, width, channels]
    channels = heads * dim

    '''
    bs, heads, h, w, dim = q.shape
    q = q * (dim ** -0.5)
    logits = tf.einsum('bhHWd,bhPQd->bhHWPQ', q, k)
    
    if absolute:
        logits += absolute_logits(q)
    
    weights = tf.reshape(logits, [-1, heads, h, w, h * w])
    weights = tf.nn.softmax(weights)
    weights = tf.reshape(weights, [-1, heads, h, w, h, w])
    
    attn_out = tf.einsum('bhHWPQ,bhPQd->bHWhd', weights, v)
    
    if fold_heads:
        attn_out = tf.reshape(attn_out, [-1, h, w, heads * dim])
    
    return attn_out

def MHSA(featuremap, 
         proj_factor, 
         heads, 
         target_dimension, 
         pos_enc_type='relative', 
         use_pos=True, 
         fold_heads=True):
    '''
    

    Parameters
    ----------
    featuremap : the output of last (convolution)layer
        shape: [batch_size, height, width, channels]
    proj_factor : reduce input dimension (channels)
        type: int 
    heads : the number of heads (mulit-head)
        type: int
    target_dimension : the output channels of MHSA layer
        type: int
    pos_enc_type : the option of positional embedding
        type: string. The default is 'relative'.
    use_pos : use positional embedding or not
        type: boolean. The default is True.
    fold_heads : combine heads or not
        type: boolean. The default is True.

    Returns
    the output shape: [batch_size, height, width, channels // proj_factor]

    '''
    q = group_pointwise(featuremap, 
                        proj_factor=proj_factor, 
                        name='q_proj', 
                        heads=heads, 
                        target_dimension=target_dimension)
    
    k = group_pointwise(featuremap, 
                        proj_factor=proj_factor, 
                        name='k_proj', 
                        heads=heads, 
                        target_dimension=target_dimension)
    
    v = group_pointwise(featuremap, 
                        proj_factor=proj_factor, 
                        name='v_proj', 
                        heads=heads, 
                        target_dimension=target_dimension)
    
    assert pos_enc_type in ['relative', 'absolute']
    if pos_enc_type == 'relative':
        o = relpos_self_attention(q=q, 
                                  k=k, 
                                  v=v, 
                                  relative=use_pos, 
                                  fold_heads=fold_heads)
    else:
        o = abspos_self_attention(q=q, 
                                  k=k, 
                                  v=v, 
                                  absolute=use_pos, 
                                  fold_heads=fold_heads)
    
    return o



