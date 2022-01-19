#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple

import torch

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 08/10/2019
           """

from torch.nn import Module

__all__ = ["MinMaxNorm", "Reshape"]


class MinMaxNorm(Module):
    """ """

    def __init__(self, min_value: float = 0, max_value: float = 1):
        """

        :param min_value:
        :param max_value:"""
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """

        :param tensor:
        :return:"""
        min_tensor = tensor.min()
        tensor -= min_tensor
        max_tensor = tensor.max()
        tensor /= max_tensor
        return tensor * (self.max_value - self.min_value) + self.min_value


class Reshape(Module):
    """
    Reshaping Layer"""

    def __init__(self, new_size: Tuple[int, ...]):
        """

        :param new_size:"""
        super().__init__()
        self.new_size = new_size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """

        :param img:
        :return:"""
        return torch.reshape(img, self.new_size)
