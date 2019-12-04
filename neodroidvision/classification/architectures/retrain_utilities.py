#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 13/11/2019
           """

from torch import nn


def set_all_parameter_requires_grad(model: nn.Module, bo: bool = False):
  for param in model.parameters():
    param.requires_grad = bo


def set_first_n_parameter_requires_grad(model: nn.Module, n: int = 6, bo: bool = False):
  for i, child in enumerate(model.children()):
    if i <= n:
      set_all_parameter_requires_grad(child, bo)
