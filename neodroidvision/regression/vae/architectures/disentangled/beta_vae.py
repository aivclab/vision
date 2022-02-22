#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple

import numpy

__author__ = "Christian Heider Nielsen"
__doc__ = ""

import torch
from torch import nn

from neodroidvision.regression.vae.architectures.vae import VAE

__all__ = ["HigginsBetaVae", "BurgessBetaVae"]


class HigginsBetaVae(VAE):
    """
    Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017).
    """

    def __init__(self, input_channels=3, latent_size=10):
        super().__init__(latent_size)

        self.inner_2d_shape = (256, 1, 1)
        self.inner_fc_shape = numpy.prod(self.inner_2d_shape).item()

        self.encoder = nn.Sequential(
            self.conv_module(input_channels, 32),
            self.conv_module(32, 32),
            self.conv_module(32, 64),
            self.conv_module(64, 64),
            self.conv_module(64, self.inner_fc_shape, stride=1, padding=0),
            self.View((-1, self.inner_fc_shape)),
        )

        self.fc_mean = nn.Linear(self.inner_fc_shape, self._latent_size)
        self.fc_std = nn.Linear(self.inner_fc_shape, self._latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(self._latent_size, self.inner_fc_shape),
            nn.ReLU(True),
            self.View((-1, *self.inner_2d_shape)),
            self.deconv_module(self.inner_fc_shape, 64, stride=1, padding=0),
            self.deconv_module(64, 64),
            self.deconv_module(64, 32),
            self.deconv_module(32, 32),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
        )

        self.weight_init()

    @staticmethod
    def conv_module(
        in_channels, out_channels, kernel_size=4, stride=2, padding=1, **conv_kwargs
    ):
        """

        Args:
          in_channels:
          out_channels:
          kernel_size:
          stride:
          padding:
          **conv_kwargs:

        Returns:

        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                **conv_kwargs
            ),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    @staticmethod
    def deconv_module(
        in_channels, out_channels, kernel_size=4, stride=2, padding=1, **convt_kwargs
    ):
        """

        Args:
          in_channels:
          out_channels:
          kernel_size:
          stride:
          padding:
          **convt_kwargs:

        Returns:

        """
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                **convt_kwargs
            ),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def encode(self, *x) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
          *x:

        Returns:

        """
        x = self.encoder(*x)
        return self.fc_mean(x), self.fc_std(x)

    def decode(self, *z) -> torch.Tensor:
        """

        Args:
          *z:

        Returns:

        """
        return self.decoder(*z)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Args:
          x:

        Returns:

        """
        mu, log_var = self.encode(x)
        return self.decode(self.reparameterise(mu, log_var)), mu, log_var


class BurgessBetaVae(HigginsBetaVae):
    """
    Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018).
    """

    def __init__(self, input_channels=3, latent_size=10):
        super().__init__(latent_size)

        self.inner_2d_shape = (32, 4, 4)
        self.inner_fc_shape = numpy.prod(self.inner_2d_shape).item()

        self.encoder = nn.Sequential(
            self.conv_module(input_channels, 32),
            self.conv_module(32, 32),
            self.conv_module(32, 32),
            self.conv_module(32, 32),
            self.View((-1, self.inner_fc_shape)),
            nn.Linear(self.inner_fc_shape, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
        )

        self.fc_mean = nn.Linear(256, self._latent_size)
        self.fc_std = nn.Linear(256, self._latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(self._latent_size, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, self.inner_fc_shape),
            nn.ReLU(True),
            self.View((-1, *self.inner_2d_shape)),
            self.deconv_module(32, 32),
            self.deconv_module(32, 32),
            self.deconv_module(32, 32),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
        )

        self.weight_init()
