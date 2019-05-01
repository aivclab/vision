#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from vision.segmentation.architectures.fcn import DownConvolution

__author__ = 'cnheider'


def fcn_encoder(in_channels, depth, start_channels):
  down_convolutions = []
  new_layer_channels = start_channels
  prev_layer_channels = in_channels
  for i in range(depth):
    pooling = True if i < depth - 1 else False
    new_layer_channels = new_layer_channels * 2
    down_conv = DownConvolution(prev_layer_channels, new_layer_channels, pooling=pooling)
    prev_layer_channels = new_layer_channels
    down_convolutions.append(down_conv)

  return down_convolutions, prev_layer_channels
