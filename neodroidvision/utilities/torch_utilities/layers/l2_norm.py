#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 04/03/2020
           """

import torch
from torch import nn
from torch.nn import init

__all__ = ["L2Norm"]


class L2Norm(nn.Module):
    """"""

    def __init__(self, n_channels: int, scale: float):
        """

        :param n_channels:
        :param scale:"""
        super().__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """

        :return:"""
        init.constant_(self.weight, self.gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:"""
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        return self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
