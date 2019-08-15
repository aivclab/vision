#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from generative.cvae.archs.vae import VAE

__author__ = 'cnheider'
__doc__ = ''

import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaVAE(VAE):

  def __init__(self, flat_image_size=256, channels=3, encoding_size=20, beta=3e-1):
    super().__init__(encoding_size=encoding_size)

    self.beta = beta
    self.flat_image_size = flat_image_size

    # encoder
    self.encoder = nn.Sequential(self._conv_module(channels, 32),
                                 self._conv_module(32, 32),
                                 self._conv_module(32, 64),
                                 self._conv_module(64, 64),
                                 )
    self.fc_mu = nn.Linear(flat_image_size, encoding_size)
    self.fc_var = nn.Linear(flat_image_size, encoding_size)

    # decoder
    self.decoder = nn.Sequential(self._deconv_module(64, 64),
                                 self._deconv_module(64, 32),
                                 self._deconv_module(32, 32, 1),
                                 self._deconv_module(32, channels),
                                 nn.Sigmoid()
                                 )
    self.fc_z = nn.Linear(encoding_size, flat_image_size)

  def encode(self, x):
    x = self.encoder(x).view(-1, self.flat_image_size)
    return self.fc_mu(x), self.fc_var(x)

  @staticmethod
  def sampler(mu, log_var):
    std = torch.exp(0.5 * log_var)  # e^(1/2 * log(std^2))
    eps = torch.randn_like(std)  # random ~ N(0, 1)
    return eps.mul(std).add_(mu)

  def decode(self, z):
    z = self.fc_z(z).view(-1, 64, 2, 2)
    return self.decoder(z)

  def forward(self, x):
    mu, log_var = self.encode(x)
    z = self.sampler(mu, log_var)
    rx = self.decode(z)
    return rx, mu, log_var

  @staticmethod
  def _conv_module(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size=4,
                                   stride=2
                                   ),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU()
                         )

  # out_padding is used to ensure output size matches EXACTLY of conv2d;
  # it does not actually add zero-padding to output :)
  @staticmethod
  def _deconv_module(in_channels, out_channels, out_padding=0):
    return nn.Sequential(nn.ConvTranspose2d(in_channels,
                                            out_channels,
                                            kernel_size=4,
                                            stride=2,
                                            output_padding=out_padding
                                            ),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU()
                         )

  def loss(self, recon_x, x, mu, log_var):
    # reconstruction losses are summed over all elements and batch
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_diverge = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return (recon_loss + self.beta * kl_diverge) / x.shape[0]  # divide total loss by batch size
