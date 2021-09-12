#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 25-01-2021
           """

import torch

__all__ = ["swish"]


def swish(x):
    """Swish activation function"""
    return x * torch.sigmoid(x)
