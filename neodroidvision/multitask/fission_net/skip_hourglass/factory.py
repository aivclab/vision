#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from neodroidvision.multitask.fission_net.skip_hourglass.compress import Compress
from neodroidvision.multitask.fission_net.skip_hourglass.decompress import Decompress

__author__ = 'Christian Heider Nielsen'
__doc__ = r'''

           Created on 29/10/2019
           '''


def fcn_encoder(in_channels: int, depth: int, start_channels: int):
  down_convolutions = []
  new_layer_channels = start_channels
  prev_layer_channels = in_channels
  for i in range(depth):
    pooling = True if i < depth - 1 else False
    new_layer_channels = new_layer_channels * 2
    down_conv = Compress(prev_layer_channels, new_layer_channels, pooling=pooling)
    prev_layer_channels = new_layer_channels
    down_convolutions.append(down_conv)

  return down_convolutions, prev_layer_channels


def fcn_decoder(in_channels: int, depth: int, up_mode: str, merge_mode: str):
  up_convolutions_ae = []
  ae_prev_layer_channels = in_channels
  for i in range(depth - 1):
    # create the decoder pathway and add to a list - careful! decoding only requires depth-1 blocks
    ae_new_layer_channels = ae_prev_layer_channels // 2
    up_conv = Decompress(ae_prev_layer_channels,
                         ae_new_layer_channels,
                         up_mode=up_mode,
                         merge_mode=merge_mode)
    ae_prev_layer_channels = ae_new_layer_channels
    up_convolutions_ae.append(up_conv)

  return up_convolutions_ae, ae_prev_layer_channels
