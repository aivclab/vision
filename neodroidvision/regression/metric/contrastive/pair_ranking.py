#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 27/07/2020
           """

__all__ = ["PairRankingSiamese"]

from typing import Tuple, Union

import torch
from draugr.torch_utilities import conv2d_hw_shape, pad2d_hw_shape
from numpy import product
from torch import nn


class PairRankingSiamese(nn.Module):
    """ """

    def __init__(
        self,
        in_size: Union[int, Tuple[int, int]] = (105, 105),
        output_size: int = 1,
        input_channels: int = 1,
    ):
        super().__init__()

        flat_lin_size = 32 * product(
            conv2d_hw_shape(
                pad2d_hw_shape(
                    conv2d_hw_shape(
                        pad2d_hw_shape(
                            conv2d_hw_shape(pad2d_hw_shape(in_size, 1), 3), 1
                        ),
                        3,
                    ),
                    1,
                ),
                3,
            )
        )

        self.concat_merge = True
        if self.concat_merge:
            flat_lin_size *= 2

        self.mapping = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_channels, 64, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=0.2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=0.2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 32, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=0.2),
            nn.Flatten(),
        )

        self.head = nn.Sequential(
            nn.Linear(flat_lin_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
            nn.Sigmoid(),
        )

    def forward(self, anchor: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        """

        Args:
          anchor:
          other:

        Returns:

        """
        if self.concat_merge:
            dist = torch.cat((self.mapping(anchor), self.mapping(other)), 1)
        else:
            dist = torch.abs(self.mapping(anchor) - self.mapping(other))
        return self.head(dist)
