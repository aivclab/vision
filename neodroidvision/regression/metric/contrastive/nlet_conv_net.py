#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 30/06/2020
           """

from typing import List

import torch
from draugr.torch_utilities import conv2d_hw_shape
from draugr.torch_utilities.operations.sizes.pad2d import pad2d_hw_shape
from numpy import product
from torch import nn

__all__ = ["NLetConvNet"]


class NLetConvNet(nn.Module):
    """ """

    def __init__(self, in_size=None, output_size: int = 2):
        super().__init__()
        flat_lin_size = 8 * product(
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

        self.convolutions = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Dropout2d(p=0.2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=0.2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=0.2),
            nn.Flatten(),
            nn.Linear(flat_lin_size, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, output_size),
        )

    def forward(self, *n_let) -> List[torch.Tensor]:
        """

        :param input1:
        :type input1:
        :param input2:
        :type input2:
        :return:
        :rtype:"""
        return [self.convolutions(x) for x in n_let]
