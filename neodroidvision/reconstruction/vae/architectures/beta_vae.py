#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple

from neodroidvision.reconstruction.vae.architectures.vae import VAE

__author__ = 'cnheider'
__doc__ = ''

import torch
import torch.nn as nn


class View(nn.Module):
  def __init__(self, size):
    super(View, self).__init__()
    self.size = size

  def forward(self, tensor):
    return tensor.view(self.size)


class BetaVAE(VAE):

  def __init__(self, channels=3, latent_size=8):
    super().__init__(encoding_size=latent_size)

    self.flat_image_size = 256

    self.encoder = nn.Sequential(nn.Conv2d(channels, 32, 4, 2),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU(True),
                                 nn.Conv2d(32, 32, 4, 2),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU(True),
                                 nn.Conv2d(32, 64, 4, 2),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(True),
                                 nn.Conv2d(64, 64, 4, 2),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(True)
                                 )

    self.fc_mu = nn.Linear(self.flat_image_size, latent_size)
    self.fc_var = nn.Linear(self.flat_image_size, latent_size)

    self.fc_z = nn.Linear(latent_size, self.flat_image_size)

    self.decoder = nn.Sequential(nn.ConvTranspose2d(64, 64, 4, 2),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(True),
                                 nn.ConvTranspose2d(64, 32, 4, 2),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU(True),
                                 nn.ConvTranspose2d(32, 32, 4, 2, output_padding=1),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU(True),
                                 nn.ConvTranspose2d(32, channels, 4, 2),
                                 nn.BatchNorm2d(channels),
                                 nn.Sigmoid()
                                 )

  def encode(self, x):
    x = self.encoder(x)
    x = x.view(-1, self.flat_image_size)
    return self.fc_mu(x), self.fc_var(x)

  @staticmethod
  def sampler(mu, log_var):
    std = torch.exp(0.5 * log_var)  # e^(1/2 * log(std^2))
    eps = torch.randn_like(std)  # random ~ N(0, 1)
    return eps.mul(std).add_(mu)  # Reparameterise distribution

  def decode(self, z):
    z = self.fc_z(z)
    z = z.view(-1, 64, 2, 2)
    return self.decoder(z)

  def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu, log_var = self.encode(x)
    z = self.sampler(mu, log_var)
    reconstruction = self.decode(z)
    return reconstruction, mu, log_var
