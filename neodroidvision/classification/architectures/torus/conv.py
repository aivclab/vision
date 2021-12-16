#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 05-03-2021
           """

__all__ = ["TorusConv2d"]

import torch


class TorusConv2d(torch.nn.Module):
    """ """

    def __init__(self, input_dim: int, output_dim: int, kernel_size, bn: bool):
        super().__init__()
        self.edge_size = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = torch.nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size)
        self.bn = torch.nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
          x:

        Returns:

        """
        h = torch.cat(
            [x[:, :, :, -self.edge_size[1] :], x, x[:, :, :, : self.edge_size[1]]],
            dim=3,
        )
        h = torch.cat(
            [h[:, :, -self.edge_size[0] :], h, h[:, :, : self.edge_size[0]]], dim=2
        )
        h = self.conv(h)
        if self.bn is not None:
            h = self.bn(h)
        return h
