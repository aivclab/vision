#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Christian Heider Nielsen'
__doc__ = r'''
           '''

# Build a Mock Model in PyTorch with a convolution and a reduceMean layer
import torch
import torch.nn as nn
import torch.onnx as torch_onnx
from torch.autograd import Variable


class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding=0, bias=False)

  def forward(self, inputs):
    x = self.conv(inputs)
    # x = x.view(x.size()[0], x.size()[1], -1)
    return torch.mean(x, dim=2)


# Use this an input trace to serialize the model
input_shape = (3, 100, 100)
model_onnx_path = "torch_model.onnx"
model = Model()
model.train(False)

# Export the model to an ONNX file
dummy_input = Variable(torch.randn(1, *input_shape))
output = torch_onnx.export(model,
                           dummy_input,
                           model_onnx_path,
                           verbose=False)
print("Export of torch_model.onnx complete!")
