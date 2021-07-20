# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 18:12:16 2021

@author: default
"""
from torch import nn

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)
    
 
