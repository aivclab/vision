#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from neodroidvision.reconstruction import VAE

__author__ = 'Christian Heider Nielsen'
__doc__ = ''

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
  def __init__(self, input_size=784, output_size=20):
    super().__init__()
    self.fcs = nn.Sequential(nn.Linear(input_size, 400), nn.ReLU(),
                             nn.Linear(400, 200), nn.ReLU())
    self.mean = nn.Linear(200, output_size)
    self.log_std = nn.Linear(200, output_size)

  def encode(self, x):
    x.view(-1, self._input_size)
    h1 = self.fcs(x)
    return self.mean(h1), self.log_std(h1)

  def forward(self, x):
    return self.encode(x)


class Decoder(nn.Module):
  def __init__(self, input_size=20, output_size=784):
    super().__init__()
    self.fcs = nn.Sequential(nn.Linear(input_size, 200), nn.ReLU(),
                             nn.Linear(200, 400), nn.ReLU(),
                             nn.Linear(400, output_size), nn.Sigmoid())

  def decode(self, z):
    h3 = self.fcs(z)
    return h3

  def forward(self, x):
    return self.decode(x).view(-1, 28, 28)


class VanillaVAE(VAE):
  def encode(self, *x: torch.Tensor) -> torch.Tensor:
    return self._encoder(*x)

  def decode(self, *x: torch.Tensor) -> torch.Tensor:
    return self._decoder(*x)

  def __init__(self, input_size=784, latent_size=2):
    super().__init__(latent_size)
    self._input_size = input_size
    self._encoder = Encoder(input_size=input_size, output_size=latent_size)
    self._decoder = Decoder(input_size=latent_size, output_size=input_size)

  def forward(self, x):
    mean, log_var = self.encode(x)
    z = self.reparameterise(mean, log_var)
    return self.decode(z), mean, log_var

  # Reconstruction + KL divergence losses summed over all elements and batch
  def loss_function(self, recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, self._input_size), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return BCE + KLD
