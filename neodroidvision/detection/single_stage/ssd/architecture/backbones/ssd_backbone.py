#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 10/11/2019
           """

from typing import Any

from torch import nn

__all__ = ["SSDBackbone"]


class SSDBackbone(nn.Module):
    """description"""

    def __init__(self, IMAGE_SIZE: Any):
        super().__init__()

    def reset_parameters(self):
        """description"""
        for m in self.extras.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
