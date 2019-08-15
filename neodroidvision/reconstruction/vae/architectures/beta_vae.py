#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple

from neodroidvision.reconstruction.vae.architectures.vae import VAE

__author__ = 'cnheider'
__doc__ = ''

import torch
import torch.nn as nn
import torch.nn.functional as F


class View(nn.Module):
  def __init__(self, size):
    super(View, self).__init__()
    self.size = size

  def forward(self, tensor):
    return tensor.view(self.size)

class BetaVAE(VAE):

  def __init__(self,  channels=3, encoding_size=20, beta=3e-1):
    super().__init__(encoding_size=encoding_size)

    self.beta = beta
    self.flat_image_size = 256

    self.encoder = nn.Sequential(
        nn.Conv2d(channels, 32, 4, 2, 1),  # B,  32, 32, 32
        nn.ReLU(True),
        nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
        nn.ReLU(True),
        nn.Conv2d(32, 32, 4, 2, 1),  # B,  32,  8,  8
        nn.ReLU(True),
        nn.Conv2d(32, 32, 4, 2, 1),  # B,  32,  4,  4
        nn.ReLU(True),
        View((-1, 32 * 4 * 4)),  # B, 512
        nn.Linear(32 * 4 * 4, 256),  # B, 256
        nn.ReLU(True),
        nn.Linear(256, 256),  # B, 256
        nn.ReLU(True)
        )

    self.fc_mu = nn.Linear(self.flat_image_size, encoding_size)
    self.fc_var = nn.Linear(self.flat_image_size, encoding_size)

    self.fc_z = nn.Linear(encoding_size, self.flat_image_size)

    self.decoder = nn.Sequential(
        nn.ReLU(True),
        nn.ConvTranspose2d(256, 64, 4),  # B,  64,  4,  4
        nn.ReLU(True),
        nn.ConvTranspose2d(64, 64, 4, 2, 1),  # B,  64,  8,  8
        nn.ReLU(True),
        nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 16, 16
        nn.ReLU(True),
        nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
        nn.ReLU(True),
        nn.ConvTranspose2d(32, channels, 4, 2, 1),  # B, nc, 64, 64,
        nn.Sigmoid()
        )

  def encode(self, x):
    x = self.encoder(x).view(-1, self.flat_image_size)
    return self.fc_mu(x), self.fc_var(x)

  @staticmethod
  def sampler(mu, log_var):
    std = torch.exp(0.5 * log_var)  # e^(1/2 * log(std^2))
    eps = torch.randn_like(std)  # random ~ N(0, 1)
    return eps.mul(std).add_(mu)

  def decode(self, z):
    z = self.fc_z(z).view(-1, self.flat_image_size, 1, 1)
    return self.decoder(z)

  def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu, log_var = self.encode(x)
    z = self.sampler(mu, log_var)
    reconstruction = self.decode(z)
    return reconstruction, mu, log_var


  def loss_function(self, reconstruction, original, mu, log_var):
    # reconstruction losses are summed over all elements and batch
    recon_loss = F.binary_cross_entropy(input=reconstruction,
                                        target=original,
                                        reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_diverge = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return (recon_loss + self.beta * kl_diverge) / original.shape[0]  # divide total loss by batch size
