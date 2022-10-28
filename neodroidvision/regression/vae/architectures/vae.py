#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn
from torch.nn.init import kaiming_normal_

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
           """

from draugr.torch_utilities import VariationalAutoEncoder

__all__ = ["VAE"]


class VAE(VariationalAutoEncoder):
    """description"""

    def __init__(self, latent_size=10):
        super().__init__()
        self._latent_size = latent_size

    class View(nn.Module):
        """description"""

        def __init__(self, size):
            super().__init__()
            self.size = size

        def forward(self, tensor):
            """

            Args:
              tensor:

            Returns:

            """
            return tensor.reshape(self.size)

    @staticmethod
    def kaiming_init(m):
        """

        Args:
          m:
        """
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.fill_(0)

    @staticmethod
    def normal_init(m, mean, std):
        """

        :param m:
        :type m:
        :param mean:
        :type mean:
        :param std:
        :type std:
        """
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            m.weight.data.normal_(mean, std)
            if m.bias.data is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.weight.data.fill_(1)
            if m.bias.data is not None:
                m.bias.data.zero_()

    def weight_init(self):
        """description"""
        for m in self.modules():
            self.kaiming_init(m)
