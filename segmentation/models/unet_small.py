#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'cnheider'

import torch
from torch import nn


class encoding_block(nn.Module):
  """
  Convolutional batch norm block with relu activation (main block used in the encoding steps)
  """

  def __init__(self, in_size, out_size, kernel_size=3, padding=0, stride=1, dilation=1, batch_norm=True,
               dropout=False):
    super().__init__()

    if batch_norm:

      # reflection padding for same size output as input (reflection padding has shown better results than
      # zero padding)
      layers = [nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(in_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride,
                          dilation=dilation),
                nn.PReLU(),
                nn.BatchNorm2d(out_size),
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(out_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride,
                          dilation=dilation),
                nn.PReLU(),
                nn.BatchNorm2d(out_size),
                ]

    else:
      layers = [nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(in_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride,
                          dilation=dilation),
                nn.PReLU(),
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(out_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride,
                          dilation=dilation),
                nn.PReLU(), ]

    if dropout:
      layers.append(nn.Dropout())

    self.encoding_block = nn.Sequential(*layers)

  def forward(self, input):

    output = self.encoding_block(input)

    return output


class decoding_block(nn.Module):
  def __init__(self, in_size, out_size, batch_norm=False, upsampling=True):
    super().__init__()

    if upsampling:
      self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                              nn.Conv2d(in_size, out_size, kernel_size=1))
    else:
      self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)

    self.conv = encoding_block(in_size, out_size, batch_norm=batch_norm)

  def forward(self, input1, input2):
    output2 = self.up(input2)
    output1 = nn.functional.upsample(input1, output2.size()[2:], mode='bilinear')
    # output1 = F.interpolate(input1, output2.size()[2:], mode='bilinear', align_corners=True, scale_factor=2)

    return self.conv(torch.cat([output1, output2], 1))


class UNetSmall(nn.Module):
  """
  Main UNet architecture
  """

  def __init__(self, out_channels=1, input_channels=3):
    super().__init__()

    self.conv1 = encoding_block(input_channels, 32)
    self.maxpool1 = nn.MaxPool2d(kernel_size=2)
    self.conv2 = encoding_block(32, 64)
    self.maxpool2 = nn.MaxPool2d(kernel_size=2)
    self.conv3 = encoding_block(64, 128)
    self.maxpool3 = nn.MaxPool2d(kernel_size=2)
    self.conv4 = encoding_block(128, 256)
    self.maxpool4 = nn.MaxPool2d(kernel_size=2)

    self.center = encoding_block(256, 512)

    self.decode4 = decoding_block(512, 256)
    self.decode3 = decoding_block(256, 128)
    self.decode2 = decoding_block(128, 64)
    self.decode1 = decoding_block(64, 32)

    self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    self.ae_decode4 = decoding_block(512, 256)
    self.ae_decode3 = decoding_block(256, 128)
    self.ae_decode2 = decoding_block(128, 64)
    self.ae_decode1 = decoding_block(64, 32)

    self.ae_final = nn.Conv2d(32, input_channels, kernel_size=1)

  def forward(self, input):
    conv1 = self.conv1(input)
    maxpool1 = self.maxpool1(conv1)
    conv2 = self.conv2(maxpool1)
    maxpool2 = self.maxpool2(conv2)
    conv3 = self.conv3(maxpool2)
    maxpool3 = self.maxpool3(conv3)
    conv4 = self.conv4(maxpool3)
    maxpool4 = self.maxpool4(conv4)

    center = self.center(maxpool4)

    decode4 = self.decode4(conv4, center)
    decode3 = self.decode3(conv3, decode4)
    decode2 = self.decode2(conv2, decode3)
    decode1 = self.decode1(conv1, decode2)

    final = nn.functional.upsample(self.final(decode1), input.size()[2:], mode='bilinear')

    ae_decode4 = self.ae_decode4(conv4, center)
    ae_decode3 = self.ae_decode3(conv3, ae_decode4)
    ae_decode2 = self.ae_decode2(conv2, ae_decode3)
    ae_decode1 = self.ae_decode1(conv1, ae_decode2)

    ae_final = nn.functional.upsample(self.final(ae_decode1), input.size()[2:], mode='bilinear')

    return final, ae_final
