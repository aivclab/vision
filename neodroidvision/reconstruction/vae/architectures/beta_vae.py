#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple

import numpy
from torch.nn.init import kaiming_normal

from neodroidagent.utilities import to_tensor

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


def kaiming_init(m):
  if isinstance(m, (nn.Linear, nn.Conv2d)):
    kaiming_normal(m.weight)
    if m.bias is not None:
      m.bias.data.fill_(0)
  elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
    m.weight.data.fill_(1)
    if m.bias is not None:
      m.bias.data.fill_(0)


class VAE(torch.nn.Module):
  def __init__(self, latent_size=10):
    super().__init__()
    self._latent_size = latent_size

  def generate(self, num=32, device='cpu'):
    z = torch.randn(num, self._latent_size).to(device)
    samples = self.decode(z).to('cpu')
    return samples

  def get_z(self, im):
    with torch.no_grad():
      mu, log_var = self.encode(im)
      z = self.sample(mu, log_var)
    return z

  @staticmethod
  def sample(mu, log_var):
    std = torch.exp(0.5 * log_var)  # e^(1/2 * log(std^2))
    eps = torch.randn_like(std)  # random ~ N(0, 1)
    z = eps.mul(std).add_(mu)  # Reparameterise distribution
    return z

  def sample_from(self, encoding, device='cpu'):
    sample = to_tensor(encoding).to(device)
    assert (sample.shape[-1] == self._encoding_size,
            f'sample.shape[-1]:{sample.shape[-1]} !='
            f' self._encoding_size:{self._encoding_size}')
    sample = self.decode(sample).to('cpu')
    return sample


class HigginsBetaVae(VAE):
  """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

  def __init__(self, input_channels=3, latent_size=10):
    super().__init__(latent_size)

    self.inner_2d_shape = (256, 1, 1)
    self.inner_fc_shape = numpy.prod(self.inner_2d_shape).item()

    self.encoder = nn.Sequential(self.conv_module(input_channels, 32),
                                 self.conv_module(32, 32),
                                 self.conv_module(32, 64),
                                 self.conv_module(64, 64),
                                 self.conv_module(64, self.inner_fc_shape, stride=1, padding=0),
                                 View((-1, self.inner_fc_shape))
                                 )

    self.fc_mean = nn.Linear(self.inner_fc_shape, self._latent_size)
    self.fc_std = nn.Linear(self.inner_fc_shape, self._latent_size)

    self.decoder = nn.Sequential(nn.Linear(self._latent_size, self.inner_fc_shape),
                                 nn.ReLU(True),
                                 View((-1, *self.inner_2d_shape)),
                                 self.deconv_module(self.inner_fc_shape, 64, stride=1, padding=0),
                                 self.deconv_module(64, 64),
                                 self.deconv_module(64, 32),
                                 self.deconv_module(32, 32),
                                 nn.ConvTranspose2d(32,
                                                    input_channels,
                                                    kernel_size=4,
                                                    stride=2,
                                                    padding=1
                                                    )
                                 )

    self.weight_init()

  def weight_init(self):
    for m in self.modules():
      kaiming_init(m)

  @staticmethod
  def conv_module(in_channels, out_channels, kernel_size=4, stride=2, padding=1, **conv_kwargs):
    return nn.Sequential(nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   **conv_kwargs
                                   ),
                         # nn.BatchNorm2d(out_channels),
                         nn.ReLU(True)
                         )

  @staticmethod
  def deconv_module(in_channels, out_channels, kernel_size=4, stride=2, padding=1, **convt_kwargs):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels,
                           out_channels,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding,
                           **convt_kwargs
                           ),
        # nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
        )

  def encode(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
    x = self.encoder(x)
    return self.fc_mean(x), self.fc_std(x)

  def decode(self, z) -> torch.Tensor:
    return torch.sigmoid(self.decoder(z))

  def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu, log_var = self.encode(x)
    z = self.sample(mu, log_var)
    reconstruction = self.decode(z)
    return reconstruction, mu, log_var


class BurgessBetaVae(HigginsBetaVae):
  """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

  def __init__(self, input_channels=3, latent_size=10):
    super().__init__(latent_size)

    self.inner_2d_shape = (32, 4, 4)
    self.inner_fc_shape = numpy.prod(self.inner_2d_shape).item()

    self.encoder = nn.Sequential(self.conv_module(input_channels, 32),
                                 self.conv_module(32, 32),
                                 self.conv_module(32, 32),
                                 self.conv_module(32, 32),
                                 View((-1, self.inner_fc_shape)),
                                 nn.Linear(self.inner_fc_shape, 256),
                                 nn.ReLU(True),
                                 nn.Linear(256, 256),
                                 nn.ReLU(True)
                                 )

    self.fc_mean = nn.Linear(256, self._latent_size)
    self.fc_std = nn.Linear(256, self._latent_size)

    self.decoder = nn.Sequential(nn.Linear(self._latent_size, 256),
                                 nn.ReLU(True),
                                 nn.Linear(256, 256),
                                 nn.ReLU(True),
                                 nn.Linear(256, self.inner_fc_shape),
                                 nn.ReLU(True),
                                 View((-1, *self.inner_2d_shape)),
                                 self.deconv_module(32, 32),
                                 self.deconv_module(32, 32),
                                 self.deconv_module(32, 32),
                                 nn.ConvTranspose2d(32,
                                                    input_channels,
                                                    kernel_size=4,
                                                    stride=2,
                                                    padding=1
                                                    )
                                 )

    self.weight_init()
