#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from neodroidvision.reconstruction.cvae.archs.vae import VAE

__author__ = 'cnheider'
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
    return self.decode(x)


class FlatNormalVAE(VAE):
  # sampler = torch.normal

  def sampler(mean, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mean + eps * std

  def __init__(self, sampling_distribution=sampler, input_size=784, encoding_size=2):
    super().__init__(input_size, encoding_size)
    self._encoder = Encoder(input_size=input_size, output_size=encoding_size)
    self._decoder = Decoder(input_size=encoding_size, output_size=input_size)
    self._sampling_distribution = sampling_distribution

  def forward(self, x):
    flat = x.view(-1, self._input_size)
    mean, log_var = self._encoder(flat)
    z = self._sampling_distribution(mean, log_var)
    return self._decoder(z), mean, log_var

  @staticmethod
  # Reconstruction + KL divergence losses summed over all elements and batch
  def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return BCE + KLD
