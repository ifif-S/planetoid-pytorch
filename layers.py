#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 12:53:39 2018

@author: ififsun
"""

from torch import sparse
import torch
import torch.nn as nn
import torch.tensor as T
import numpy as np


EXP_SOFTMAX = True

class SparseLayer():

    def __init__(self, incoming, num_units, W , b = nn.init.constant(0.), nonlinearity = nn.ReLU, **kwargs):
        super(SparseLayer, self).__init__(incoming, **kwargs)

        self.num_units = num_units
        self.nonlinearity = nonlinearity

        num_inputs = int(np.prod(self.input_shape[1:]))

        self.W = nn.init.xavier_uniform(torch.Tensor(num_inputs, num_units),gain=1) 
        if b is None:
            self.b = None
        else:
            self.b = torch.Tensor(b, dtype=torch.float64)

    def get_output_for(self, input, **kwargs):
        act = torch.mm(input, self.W)
        if self.b is not None:
            act += torch.unsqueeze(self.b, 0)
        if not EXP_SOFTMAX or self.nonlinearity != nn.softmax:
            return self.nonlinearity(act)
        else:
            return torch.exp(act) / (torch.exp(act).sum(1, keepdims = True))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)


class NeuralNet(nn.Module):
    def __init__(self, incoming, num_units, nonlinearity = nn.ReLU, **kwargs):
        super(NeuralNet, self).__init__()
        self.nonlinearity=nonlinearity
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.num_units = num_units
        self.fc1 = nn.Linear(num_inputs, self.num_units) 
        if not EXP_SOFTMAX or self.nonlinearity != nn.softmax:
            self.nonlinearity = nonlinearity
        else:
            self.nonlinearity = nn.softmax()  
    
    def forward(self, incoming):
        x=incoming
        out = self.fc1(x)
        out = self.nonlinearity(out)
        return out
    
