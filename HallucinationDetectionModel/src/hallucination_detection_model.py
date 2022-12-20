#!/usr/bin/python3
"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0
"""
from collections import OrderedDict
import torch
from torch import nn
import json

class HallucinationDetectionModel(nn.Module):
    def __init__(self, model_input: int, first_layer_dim: int, second_layer_dim: int):
        super().__init__()

        self.net: nn.Sequential = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(model_input, first_layer_dim)),
            ("batch_norm_fc1", nn.BatchNorm1d(first_layer_dim)),
            ('relu_fc1', nn.ReLU()),
            ("dropout_fc1", nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(first_layer_dim, second_layer_dim)),
            ("batch_norm_fc2", nn.BatchNorm1d(second_layer_dim)),
            ('relu_fc2', nn.ReLU()),
            ("dropout_fc2", nn.Dropout(p=0.5)),
            ('classifier', nn.Linear(second_layer_dim, 2)),
        ]))

    def forward(self, features):

        return self.net(features)






if __name__ == "__main__":
    input_vector = torch.zeros(1000, 72)
    model = HallucinationDetectionModel(model_input=72, first_layer_dim=1024, second_layer_dim=128)
    print(model(input_vector, dump=True).shape)


