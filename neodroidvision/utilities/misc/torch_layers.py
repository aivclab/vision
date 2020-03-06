#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 08/10/2019
           """

from torch.nn import Module

__all__ = ["MinMaxNorm", "Reshape"]


class MinMaxNorm(Module):
    def __init__(self, min_value=0, max_value=1):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, tensor):
        min_tensor = tensor.min()
        tensor = tensor - min_tensor
        max_tensor = tensor.max()
        tensor = tensor / max_tensor
        tensor = tensor * (self.max_value - self.min_value) + self.min_value
        return tensor


class Reshape(Module):
    """
  Reshaping Layer
  """

    def __init__(self, new_size):
        super().__init__()
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)
