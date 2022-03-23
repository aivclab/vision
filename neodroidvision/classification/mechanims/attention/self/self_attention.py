#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
           """

from torch.nn.init import xavier_uniform_
from classification.mechanims.attention.self.spectral_norm import spectral_norm_conv2d


def init_weights(m):
    """

    Args:
      m:
    """
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


class SelfAttentionModule(nn.Module):
    """Self attention Layer"""

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.spectral_norm_conv1x1_theta = spectral_norm_conv2d(
            in_channels=in_channels,
            out_channels=in_channels // 8,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.spectral_norm_conv1x1_phi = spectral_norm_conv2d(
            in_channels=in_channels,
            out_channels=in_channels // 8,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.spectral_norm_conv1x1_g = spectral_norm_conv2d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.spectral_norm_conv1x1_attn = spectral_norm_conv2d(
            in_channels=in_channels // 2,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax = nn.Softmax(
            dim=-1
        )  # TODO: use log_softmax?, Check dim maybe it should be 1

        self.sigma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        """
        inputs :
        x : input feature maps(B X C X W X H)
        returns :
        out : self attention value + input feature
        attention: B X N X N (N is Width*Height)"""
        _, ch, h, w = x.size()
        # Theta path
        theta = self.spectral_norm_conv1x1_theta(x)
        theta = theta.reshape(-1, ch // 8, h * w)
        # Phi path
        phi = self.spectral_norm_conv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.reshape(-1, ch // 8, h * w // 4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.spectral_norm_conv1x1_g(x)
        g = self.maxpool(g)
        g = g.reshape(-1, ch // 2, h * w // 4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.reshape(-1, ch // 2, h, w)
        attn_g = self.spectral_norm_conv1x1_attn(attn_g)
        # Out
        out = x + self.sigma * attn_g
        return out
