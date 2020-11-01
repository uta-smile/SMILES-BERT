# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import torch.nn as nn


class PredNet(nn.Module):
    """PredNet mapping from fingerprint to property/properties prediction.
    """

    def __init__(self,
                 input_dim,
                 hidden_dim=512,
                 act_func='LeakyReLU',
                 num_props=1,
                 dropout=0.5,
                 cls_index=[]):
        super(PredNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_props = num_props
        self.cls_index = cls_index
        self.act_func = getattr(nn, act_func)()
        self.dropout = nn.Dropout(p=dropout) if dropout else lambda x: x
        # Multi heads with own FC.
        self.fc1 = nn.ModuleList()
        self.fc2 = nn.ModuleList()
        for i in range(num_props):
            output_dim = 2 if i in cls_index else 1
            if hidden_dim > 0:
                self.fc1.append(nn.Linear(input_dim, hidden_dim))
                self.fc2.append(nn.Linear(hidden_dim, output_dim))
            else:
                self.fc1.append(nn.Linear(input_dim, output_dim))

    def forward(self, x):
        output = []
        for i in range(self.num_props):
            x_output = self.fc1[i](x)
            if self.hidden_dim > 0:
                x_output = self.dropout(self.act_func(x_output))
                x_output = self.fc2[i](x_output)
            output.append(x_output)
        return output
