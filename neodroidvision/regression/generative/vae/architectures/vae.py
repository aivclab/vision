#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import abstractmethod

import torch
from torch import nn
from torch.nn.init import kaiming_normal_

from neodroidagent.utilities import to_tensor

__author__ = 'Christian Heider Nielsen'
__doc__ = r'''
           '''


class VAE(torch.nn.Module):
  class View(nn.Module):
    def __init__(self, size):
      super().__init__()
      self.size = size

    def forward(self, tensor):
      return tensor.view(self.size)

  @staticmethod
  def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
      kaiming_normal_(m.weight)
      if m.bias is not None:
        m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
      m.weight.data.fill_(1)
      if m.bias is not None:
        m.bias.data.fill_(0)

  def weight_init(self):
    for m in self.modules():
      self.kaiming_init(m)

  def __init__(self, latent_size=10):
    super().__init__()
    self._latent_size = latent_size

  @abstractmethod
  def encode(self, *x: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError

  @abstractmethod
  def decode(self, *x: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError

  def sample(self, *x, num=1) -> torch.Tensor:
    z = torch.randn(num, self._latent_size).to(device=next(self.parameters()).device)
    samples = self.decode(z, *x).to('cpu')
    return samples

  @staticmethod
  def reparameterise(mean, log_var) -> torch.Tensor:
    std = torch.exp(0.5 * log_var)  # e^(1/2 * log(std^2))
    eps = torch.randn_like(std)  # random ~ N(0, 1)
    z = eps.mul(std).add_(mean)  # Reparameterise distribution
    return z

  def sample_from(self, *encoding) -> torch.Tensor:
    sample = to_tensor(*encoding).to(device=next(self.parameters()).device)
    assert sample.shape[-1] == self._latent_size, (
      f'sample.shape[-1]:{sample.shape[-1]} !='
      f' self._encoding_size:{self._latent_size}')
    sample = self.decode(*sample).to('cpu')
    return sample
