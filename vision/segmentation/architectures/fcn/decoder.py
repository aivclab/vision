#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from vision.segmentation.architectures.fcn import UpConvolution

__author__ = 'cnheider'


def fcn_decoder(in_channels, depth, up_mode, merge_mode):
  up_convolutions_ae = []
  ae_prev_layer_channels = in_channels
  for i in range(depth - 1):
    # create the decoder pathway and add to a list - careful! decoding only requires depth-1 blocks
    ae_new_layer_channels = ae_prev_layer_channels // 2
    up_conv = UpConvolution(ae_prev_layer_channels,
                            ae_new_layer_channels,
                            up_mode=up_mode,
                            merge_mode=merge_mode)
    ae_prev_layer_channels = ae_new_layer_channels
    up_convolutions_ae.append(up_conv)

  return up_convolutions_ae, ae_prev_layer_channels
