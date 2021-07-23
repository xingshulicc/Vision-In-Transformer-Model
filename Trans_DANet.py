# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Tue Jul 20 15:01:01 2021

@author: default
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from BoT_Block_Layer_Torch import BottleBlock


def _SplitChannels(channels, num_groups):
    split_channels = [channels // num_groups for _ in range(num_groups)]
    # if the channels // num_groups has remainder and then the remainder will be added into the first element
    split_channels[0] += channels - sum(split_channels)
    # the split_channels is a list '[...]'
    return split_channels

def tensor_list_add(list_length, tensor_list):
    for i in range(list_length):
        if i == 0:
            out_tensor = tensor_list[i]
        else:
            out_tensor += tensor_list[i]
    
    return out_tensor 

def _calculate_out_channels(input_channels, increase_num, num_layers):
    for i in range(num_layers):
        output_channels = input_channels + increase_num
        output_channels = input_channels + output_channels
        input_channels = output_channels
    return output_channels

'''
kernel_size is a list: [1, 3, 5, ...]
padding is a list: [0, 1, 2, ...]
if dilation is 1 then padding is equal to (k - 1) / 2
how to calculate output feature map size:
    o = [i + 2*p - k - (k - 1) *(d - 1)]/s + 1
o is output
i is input
p is padding
k is kernel_size
s is stride
d is dilation
'''

class GroupedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(GroupedConv2d, self).__init__()
        
        self.split_in_channels = _SplitChannels(in_channels, num_groups)
        self.split_out_channels = _SplitChannels(out_channels, num_groups)
    
        self.num_convs_per_group = len(kernel_size)
        self.padding_size = [int((k - 1)/2) for k in kernel_size]
        
        self.per_group_convs = nn.ModuleList()
        self.grouped_conv = nn.ModuleList()
    
        for i in range(num_groups):
            for j in range(self.num_convs_per_group):
                self.per_group_convs.append(nn.Sequential(
                    nn.Conv2d(
                    self.split_in_channels[i], 
                    self.split_out_channels[i], 
                    kernel_size[j], 
                    stride=1, 
                    padding=self.padding_size[j], 
                    dilation=1, 
                    groups=1, 
                    bias=False
                    ), 
                    nn.BatchNorm2d(self.split_out_channels[i]), 
                    nn.ReLU(inplace=False)
                    ))
            self.grouped_conv.append(self.per_group_convs)
            
        self.cross_group_fusion = nn.Sequential(
            nn.Conv2d(
                out_channels, 
                out_channels,
                kernel_size=1, 
                stride=1, 
                padding=0, 
                bias=False
                ), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=False), 
            nn.Conv2d(
                out_channels, 
                out_channels,
                kernel_size=1, 
                stride=1, 
                padding=0, 
                bias=False
                ), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=False)
            )
        
    def forward(self, x):
        x_split = torch.split(x, self.split_in_channels, dim=1)
        x_cat = []
        for conv, t in zip(self.grouped_conv, x_split):
            x_i = []
            for conv_i in conv:
                x_i.append(conv_i(t))
            x_i = SK_F(conv_i(t).size(1), self.num_convs_per_group)(x_i)
            x_i = tensor_list_add(self.num_convs_per_group, x_i)
            
            x_cat.append(x_i)
            
        x_cat = torch.cat(x_cat, dim=1)
        x_cat = self.cross_group_fusion(x_cat)
        
        if x.size(1) == x_cat.size(1):
            x_cat += x
        else:
            x_cat = torch.cat([x, x_cat], dim=1)

        return x_cat
        
class SK_F(nn.Module):
    def __init__(self, input_channels, num_branches, ratio=4):
        '''
        num_branches : int, the number of branches in each group 
        ratio : int, the reduction ratio in the first hidden layer default is 4

        '''
        super(SK_F, self).__init__()
        
        self.in_channels = input_channels 
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.drop_neurons = nn.Dropout(p=0.25, inplace=False)
        
        self.share_layer = nn.Sequential(
            nn.Linear(
                input_channels, 
                input_channels // ratio, 
                bias = False
                ), 
            nn.ReLU(inplace=False)
            )
        
        self.branch_dense_layers = nn.ModuleList()
        for i in range(num_branches):
            self.branch_dense_layers.append(nn.Sequential(
                nn.Linear(
                    input_channels // ratio, 
                    input_channels, 
                    bias = False
                    ), 
                nn.Sigmoid()
                ))
        
    def forward(self, x):
        share_out = []
        branch_out = []
        final_output = []
        for x_j in x:
            x_j = self.gap(x_j)
            x_j = x_j.view(-1, self.in_channels)
            
            x_j = self.share_layer(x_j)
            x_j = self.drop_neurons(x_j)
            share_out.append(x_j)
        for out, dense in zip(share_out, self.branch_dense_layers):
            out = dense(out)
            branch_out.append(out)
        for features, branch in zip(x, branch_out):
            branch = branch.view(-1, self.in_channels, 1, 1)
            features = features * branch
            final_output.append(features)
        
        return final_output
            
class DANet(nn.Module):
    def __init__(self, building_block, num_blocks, num_classes):
        super(DANet, self).__init__()
        
        self.increase_channels = 32
        self.in_channels = 32
        # the num_layers is the number of _make_layer, here it is 3
        self.final_conv_out_channels = _calculate_out_channels(self.in_channels, self.increase_channels, num_layers=3)
        
        self.conv1 = nn.Conv2d(in_channels=3, 
                               out_channels=32, 
                               kernel_size=7, 
                               stride=2, 
                               padding=3, 
                               dilation=1, 
                               groups=1, 
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)
        self.maxpool_op = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        
        self.layer1 = self._make_layer(building_block, 
                                       kernel_size=[1, 3, 5], 
                                       num_groups=4, 
                                       num_blocks=num_blocks[0])
        self.layer2 = self._make_layer(building_block, 
                                       kernel_size=[1, 3, 5], 
                                       num_groups=4, 
                                       num_blocks=num_blocks[1])
        self.layer3 = self._make_layer(building_block, 
                                       kernel_size=[1, 3, 5], 
                                       num_groups=4, 
                                       num_blocks=num_blocks[2])
        
        # insert Transformer block for global self-attention
        self.Transformer1 = BottleBlock(dim=self.final_conv_out_channels, 
                                        fmap_size=(7, 7), 
                                        dim_out=self.final_conv_out_channels, 
                                        proj_factor=8, 
                                        stride=1, 
                                        heads=8, 
                                        dim_head=60, 
                                        rel_pos_emb=True, 
                                        activation=nn.GELU())
        
        self.linear = nn.Linear(in_features = self.final_conv_out_channels, 
                                out_features = num_classes, 
                                bias=False)
        
    def _make_layer(self, building_block, kernel_size, num_groups, num_blocks):
        assert num_blocks > 1, 'the input value of num_blocks should be greater than 1'
        out_channels = self.in_channels + self.increase_channels
        layers = []
        for i in range(num_blocks - 1):
            layers.append(building_block(
                self.in_channels, 
                self.in_channels, 
                kernel_size, 
                num_groups
                ))
        layers.append(building_block(
            self.in_channels, 
            out_channels, 
            kernel_size, 
            num_groups
            ))
        layers.append(self.maxpool_op)
        self.in_channels += out_channels
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool_1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.Transformer1(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.linear(out)
        
        return out


num_classes = 10
def DANet24():
    return DANet(GroupedConv2d, [2, 2, 3], num_classes)



        
    
    
