#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from neodroidvision.mechanims.attention.self_attention import (
  SelfAttentionModule,
  init_weights,
  spectral_norm_conv2d,
  spectral_norm_embedding,
  spectral_norm_linear,
  )

__author__ = 'Christian Heider Nielsen'
__doc__ = r'''
           '''


class ConditionalBatchNorm2d(nn.Module):
  '''https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775'''

  def __init__(self, num_features, num_classes):
    super().__init__()
    self.num_features = num_features
    self.bn = nn.BatchNorm2d(num_features, momentum=0.001, affine=False)
    self.embed = nn.Embedding(num_classes, num_features * 2)
    # self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
    self.embed.weight.data[:, :num_features].fill_(1.)  # Initialize scale to 1
    self.embed.weight.data[:, num_features:].zero_()  # Initialize bias at 0

  def forward(self, x, y):
    out = self.bn(x)
    gamma, beta = self.embed(y).chunk(2, 1)
    out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
    return out


class GenBlock(nn.Module):
  def __init__(self, in_channels, out_channels, num_classes):
    super(GenBlock, self).__init__()
    self.cond_bn1 = ConditionalBatchNorm2d(in_channels, num_classes)
    self.relu = nn.ReLU(inplace=True)
    self.spectral_norm_conv2d1 = spectral_norm_conv2d(in_channels=in_channels,
                                                      out_channels=out_channels,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1)
    self.cond_bn2 = ConditionalBatchNorm2d(out_channels,
                                           num_classes)
    self.spectral_norm_conv2d2 = spectral_norm_conv2d(in_channels=out_channels,
                                                      out_channels=out_channels,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1)
    self.spectral_norm_conv2d0 = spectral_norm_conv2d(in_channels=in_channels,
                                                      out_channels=out_channels,
                                                      kernel_size=1,
                                                      stride=1,
                                                      padding=0)

  def forward(self, x, labels):
    x0 = x

    x = self.cond_bn1(x, labels)
    x = self.relu(x)
    x = F.interpolate(x, scale_factor=2, mode='nearest')  # upsample
    x = self.spectral_norm_conv2d1(x)
    x = self.cond_bn2(x, labels)
    x = self.relu(x)
    x = self.spectral_norm_conv2d2(x)

    x0 = F.interpolate(x0, scale_factor=2, mode='nearest')  # upsample
    x0 = self.spectral_norm_conv2d0(x0)

    out = x + x0
    return out


class Generator(nn.Module):
  """Generator."""

  def __init__(self, z_dim, g_conv_dim, num_classes):
    super(Generator, self).__init__()

    self.z_dim = z_dim
    self.g_conv_dim = g_conv_dim
    self.snlinear0 = spectral_norm_linear(in_features=z_dim, out_features=g_conv_dim * 16 * 4 * 4)
    self.block1 = GenBlock(g_conv_dim * 16, g_conv_dim * 16, num_classes)
    self.block2 = GenBlock(g_conv_dim * 16, g_conv_dim * 8, num_classes)
    self.block3 = GenBlock(g_conv_dim * 8, g_conv_dim * 4, num_classes)
    self.self_attn = SelfAttentionModule(g_conv_dim * 4)
    self.block4 = GenBlock(g_conv_dim * 4, g_conv_dim * 2, num_classes)
    self.block5 = GenBlock(g_conv_dim * 2, g_conv_dim, num_classes)
    self.bn = nn.BatchNorm2d(g_conv_dim, eps=1e-5, momentum=0.0001, affine=True)
    self.relu = nn.ReLU(inplace=True)
    self.spectral_norm_conv2d1 = spectral_norm_conv2d(in_channels=g_conv_dim, out_channels=3, kernel_size=3,
                                                      stride=1, padding=1)
    self.tanh = nn.Tanh()

    # Weight init
    self.apply(init_weights)

  def forward(self, z, labels):
    # n x z_dim
    act0 = self.snlinear0(z)  # n x g_conv_dim*16*4*4
    act0 = act0.view(-1, self.g_conv_dim * 16, 4, 4)  # n x g_conv_dim*16 x 4 x 4
    act1 = self.block1(act0, labels)  # n x g_conv_dim*16 x 8 x 8
    act2 = self.block2(act1, labels)  # n x g_conv_dim*8 x 16 x 16
    act3 = self.block3(act2, labels)  # n x g_conv_dim*4 x 32 x 32
    act3 = self.self_attn(act3)  # n x g_conv_dim*4 x 32 x 32
    act4 = self.block4(act3, labels)  # n x g_conv_dim*2 x 64 x 64
    act5 = self.block5(act4, labels)  # n x g_conv_dim  x 128 x 128
    act5 = self.bn(act5)  # n x g_conv_dim  x 128 x 128
    act5 = self.relu(act5)  # n x g_conv_dim  x 128 x 128
    act6 = self.spectral_norm_conv2d1(act5)  # n x 3 x 128 x 128
    act6 = self.tanh(act6)  # n x 3 x 128 x 128
    return act6


class DiscOptBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(DiscOptBlock, self).__init__()
    self.spectral_norm_conv2d1 = spectral_norm_conv2d(in_channels=in_channels,
                                                      out_channels=out_channels,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1)
    self.relu = nn.ReLU(inplace=True)
    self.spectral_norm_conv2d2 = spectral_norm_conv2d(in_channels=out_channels,
                                                      out_channels=out_channels,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1)
    self.downsample = nn.AvgPool2d(2)
    self.spectral_norm_conv2d0 = spectral_norm_conv2d(in_channels=in_channels,
                                                      out_channels=out_channels,
                                                      kernel_size=1,
                                                      stride=1,
                                                      padding=0)

  def forward(self, x):
    x0 = x

    x = self.spectral_norm_conv2d1(x)
    x = self.relu(x)
    x = self.spectral_norm_conv2d2(x)
    x = self.downsample(x)

    x0 = self.downsample(x0)
    x0 = self.spectral_norm_conv2d0(x0)

    out = x + x0
    return out


class DiscBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(DiscBlock, self).__init__()
    self.relu = nn.ReLU(inplace=True)
    self.spectral_norm_conv2d1 = spectral_norm_conv2d(in_channels=in_channels,
                                                      out_channels=out_channels,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1)
    self.spectral_norm_conv2d2 = spectral_norm_conv2d(in_channels=out_channels,
                                                      out_channels=out_channels,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1)
    self.downsample = nn.AvgPool2d(2)
    self.ch_mismatch = False
    if in_channels != out_channels:
      self.ch_mismatch = True
    self.spectral_norm_conv2d0 = spectral_norm_conv2d(in_channels=in_channels,
                                                      out_channels=out_channels,
                                                      kernel_size=1,
                                                      stride=1,
                                                      padding=0)

  def forward(self, x, downsample=True):
    x0 = x

    x = self.relu(x)
    x = self.spectral_norm_conv2d1(x)
    x = self.relu(x)
    x = self.spectral_norm_conv2d2(x)
    if downsample:
      x = self.downsample(x)

    if downsample or self.ch_mismatch:
      x0 = self.spectral_norm_conv2d0(x0)
      if downsample:
        x0 = self.downsample(x0)

    out = x + x0
    return out


class Discriminator(nn.Module):
  """Discriminator."""

  def __init__(self, d_conv_dim, num_classes):
    super(Discriminator, self).__init__()
    self.d_conv_dim = d_conv_dim
    self.opt_block1 = DiscOptBlock(3, d_conv_dim)
    self.block1 = DiscBlock(d_conv_dim, d_conv_dim * 2)
    self.self_attn = SelfAttentionModule(d_conv_dim * 2)
    self.block2 = DiscBlock(d_conv_dim * 2, d_conv_dim * 4)
    self.block3 = DiscBlock(d_conv_dim * 4, d_conv_dim * 8)
    self.block4 = DiscBlock(d_conv_dim * 8, d_conv_dim * 16)
    self.block5 = DiscBlock(d_conv_dim * 16, d_conv_dim * 16)
    self.relu = nn.ReLU(inplace=True)
    self.snlinear1 = spectral_norm_linear(in_features=d_conv_dim * 16, out_features=1)
    self.sn_embedding1 = spectral_norm_embedding(num_classes, d_conv_dim * 16)

    # Weight init
    self.apply(init_weights)
    xavier_uniform_(self.sn_embedding1.weight)

  def forward(self, x, labels):
    # n x 3 x 128 x 128
    h0 = self.opt_block1(x)  # n x d_conv_dim   x 64 x 64
    h1 = self.block1(h0)  # n x d_conv_dim*2 x 32 x 32
    h1 = self.self_attn(h1)  # n x d_conv_dim*2 x 32 x 32
    h2 = self.block2(h1)  # n x d_conv_dim*4 x 16 x 16
    h3 = self.block3(h2)  # n x d_conv_dim*8 x  8 x  8
    h4 = self.block4(h3)  # n x d_conv_dim*16 x 4 x  4
    h5 = self.block5(h4, downsample=False)  # n x d_conv_dim*16 x 4 x 4
    h5 = self.relu(h5)  # n x d_conv_dim*16 x 4 x 4
    h6 = torch.sum(h5, dim=[2, 3])  # n x d_conv_dim*16
    output1 = torch.squeeze(self.snlinear1(h6))  # n x 1
    # Projection
    h_labels = self.sn_embedding1(labels)  # n x d_conv_dim*16
    proj = torch.mul(h6, h_labels)  # n x d_conv_dim*16
    output2 = torch.sum(proj, dim=[1])  # n x 1
    # Out
    output = output1 + output2  # n x 1
    return output
