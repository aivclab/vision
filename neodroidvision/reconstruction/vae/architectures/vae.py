#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

from neodroidagent.utilities import to_tensor

__author__ = 'cnheider'
__doc__ = ''


class VAE(torch.nn.Module):
  def __init__(self, input_size=784, encoding_size=3):
    super().__init__()
    self._input_size = input_size
    self._encoding_size = encoding_size

  def sample(self, num=64, device='cpu'):
    sample = torch.randn(num, self._encoding_size).to(device)
    sample = self.decode(sample).to('cpu')
    return sample

  def sample_from(self, encoding, device='cpu'):
    sample = to_tensor(encoding).to(device)
    assert (sample.shape[-1] == self._encoding_size,
            f'sample.shape[-1]:{sample.shape[-1]} !='
            f' self._encoding_size:{self._encoding_size}')
    sample = self.decode(sample).to('cpu')
    return sample
