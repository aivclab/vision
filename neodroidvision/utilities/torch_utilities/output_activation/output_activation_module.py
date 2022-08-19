#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 03/06/2020
           """

__all__ = ["OutputActivationModule"]

from typing import Tuple, Union

import torch
from torch.nn import Module


class OutputActivationModule(Module):
    """
    For adding output activation to traced torchscript architectures"""

    def __init__(self, model: Module, output_activation: callable = torch.sigmoid):
        super().__init__()
        self._model = model
        self._output_activation = output_activation

    def forward(self, *args, **kwargs) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        """

        Args:
          *args:
          **kwargs:

        Returns:

        """
        out = self._model(*args, **kwargs)
        if isinstance(out, tuple):
            return (*[self._output_activation(a) for a in out],)
        return self._output_activation(out)
