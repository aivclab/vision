#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from classification.mechanims.attention import (
    SelfAttentionModule,
    init_weights,
)
from classification.mechanims.attention.self.spectral_norm import (
    spectral_norm_conv2d,
    spectral_norm_linear,
    spectral_norm_embedding,
)
from torch import nn
from torch.nn import functional
from torch.nn.init import xavier_uniform_

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
           """


class ConditionalBatchNorm2d(nn.Module):
    """https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775"""

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, momentum=0.001, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        # self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, :num_features].fill_(1.0)  # Initialize scale to 1
        self.embed.weight.data[:, num_features:].zero_()  # Initialize bias at 0

    def forward(self, x, y):
        """

        Args:
          x:
          y:

        Returns:

        """
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        return gamma.reshape(-1, self.num_features, 1, 1) * out + beta.reshape(
            -1, self.num_features, 1, 1
        )


class GenBlock(nn.Module):
    """ """

    def __init__(self, in_channels, out_channels, num_classes):
        super(GenBlock, self).__init__()
        self.cond_bn1 = ConditionalBatchNorm2d(in_channels, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.spectral_norm_conv2d1 = spectral_norm_conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.cond_bn2 = ConditionalBatchNorm2d(out_channels, num_classes)
        self.spectral_norm_conv2d2 = spectral_norm_conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.spectral_norm_conv2d0 = spectral_norm_conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x, labels):
        """

        Args:
          x:
          labels:

        Returns:

        """
        x0 = x

        x = self.cond_bn1(x, labels)
        x = self.relu(x)
        x = functional.interpolate(x, scale_factor=2, mode="nearest")  # upsample
        x = self.spectral_norm_conv2d1(x)
        x = self.cond_bn2(x, labels)
        x = self.relu(x)
        x = self.spectral_norm_conv2d2(x)

        x0 = functional.interpolate(x0, scale_factor=2, mode="nearest")  # upsample
        x0 = self.spectral_norm_conv2d0(x0)

        return x + x0


class Generator(nn.Module):
    """Generator."""

    def __init__(self, z_dim, g_conv_dim, num_classes):
        super(Generator, self).__init__()

        self.z_dim = z_dim
        self.g_conv_dim = g_conv_dim
        self.spectral_norm_linear0 = spectral_norm_linear(
            in_features=z_dim, out_features=g_conv_dim * 16 * 4 * 4
        )
        self.block1 = GenBlock(g_conv_dim * 16, g_conv_dim * 16, num_classes)
        self.block2 = GenBlock(g_conv_dim * 16, g_conv_dim * 8, num_classes)
        self.block3 = GenBlock(g_conv_dim * 8, g_conv_dim * 4, num_classes)
        self.self_attn = SelfAttentionModule(g_conv_dim * 4)
        self.block4 = GenBlock(g_conv_dim * 4, g_conv_dim * 2, num_classes)
        self.block5 = GenBlock(g_conv_dim * 2, g_conv_dim, num_classes)
        self.bn = nn.BatchNorm2d(g_conv_dim, eps=1e-5, momentum=0.0001, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.spectral_norm_conv2d1 = spectral_norm_conv2d(
            in_channels=g_conv_dim, out_channels=3, kernel_size=3, stride=1, padding=1
        )
        self.tanh = nn.Tanh()

        self.apply(init_weights)

    def forward(self, z, labels):
        """

        Args:
          z:
          labels:

        Returns:

        """
        # n x z_dim
        act0 = self.spectral_norm_linear0(z)  # n x g_conv_dim*16*4*4
        act0 = act0.reshape(-1, self.g_conv_dim * 16, 4, 4)  # n x g_conv_dim*16 x 4 x 4
        act1 = self.block1(act0, labels)  # n x g_conv_dim*16 x 8 x 8
        act2 = self.block2(act1, labels)  # n x g_conv_dim*8 x 16 x 16
        act3 = self.block3(act2, labels)  # n x g_conv_dim*4 x 32 x 32
        act3 = self.self_attn(act3)  # n x g_conv_dim*4 x 32 x 32
        act4 = self.block4(act3, labels)  # n x g_conv_dim*2 x 64 x 64
        act5 = self.block5(act4, labels)  # n x g_conv_dim  x 128 x 128
        act5 = self.bn(act5)  # n x g_conv_dim  x 128 x 128
        act5 = self.relu(act5)  # n x g_conv_dim  x 128 x 128
        act6 = self.spectral_norm_conv2d1(act5)  # n x 3 x 128 x 128
        return self.tanh(act6)  # n x 3 x 128 x 128


class DiscriminatorOptBlock(nn.Module):
    """ """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.spectral_norm_conv2d1 = spectral_norm_conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.relu = nn.ReLU(inplace=True)
        self.spectral_norm_conv2d2 = spectral_norm_conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.down_sample = nn.AvgPool2d(2)
        self.spectral_norm_conv2d0 = spectral_norm_conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        """

        Args:
          x:

        Returns:

        """
        x0 = x

        x = self.spectral_norm_conv2d1(x)
        x = self.relu(x)
        x = self.spectral_norm_conv2d2(x)
        x = self.down_sample(x)

        x0 = self.down_sample(x0)
        x0 = self.spectral_norm_conv2d0(x0)

        return x + x0


class DiscriminatorBlock(nn.Module):
    """ """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.spectral_norm_conv2d1 = spectral_norm_conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.spectral_norm_conv2d2 = spectral_norm_conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.down_sample = nn.AvgPool2d(2)
        self.channel_mismatch = False
        if in_channels != out_channels:
            self.channel_mismatch = True
        self.spectral_norm_conv2d0 = spectral_norm_conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x, down_sample: bool = True):
        """

        Args:
          x:
          down_sample:

        Returns:

        """
        x0 = x

        x = self.relu(x)
        x = self.spectral_norm_conv2d1(x)
        x = self.relu(x)
        x = self.spectral_norm_conv2d2(x)
        if down_sample:
            x = self.down_sample(x)

        if down_sample or self.channel_mismatch:
            x0 = self.spectral_norm_conv2d0(x0)
            if down_sample:
                x0 = self.down_sample(x0)

        return x + x0


class Discriminator(nn.Module):
    """Discriminator."""

    def __init__(self, d_conv_dim, num_classes):
        super(Discriminator, self).__init__()
        self.d_conv_dim = d_conv_dim
        self.opt_block1 = DiscriminatorOptBlock(3, d_conv_dim)
        self.block1 = DiscriminatorBlock(d_conv_dim, d_conv_dim * 2)
        self.self_attn = SelfAttentionModule(d_conv_dim * 2)
        self.block2 = DiscriminatorBlock(d_conv_dim * 2, d_conv_dim * 4)
        self.block3 = DiscriminatorBlock(d_conv_dim * 4, d_conv_dim * 8)
        self.block4 = DiscriminatorBlock(d_conv_dim * 8, d_conv_dim * 16)
        self.block5 = DiscriminatorBlock(d_conv_dim * 16, d_conv_dim * 16)
        self.relu = nn.ReLU(inplace=True)
        self.spectral_norm_linear1 = spectral_norm_linear(
            in_features=d_conv_dim * 16, out_features=1
        )
        self.spectral_norm_embedding1 = spectral_norm_embedding(
            num_classes, d_conv_dim * 16
        )

        self.apply(init_weights)
        xavier_uniform_(self.spectral_norm_embedding1.weight)

    def forward(self, x, labels):
        """

        Args:
          x:
          labels:

        Returns:

        """
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
        output1 = torch.squeeze(self.spectral_norm_linear1(h6))  # n x 1
        # Projection
        h_labels = self.spectral_norm_embedding1(labels)  # n x d_conv_dim*16
        proj = torch.mul(h6, h_labels)  # n x d_conv_dim*16
        output2 = torch.sum(proj, dim=[1])  # n x 1
        return output1 + output2  # n x 1
