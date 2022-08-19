#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
           """

from typing import Iterable

import numpy
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn

one_hot_encoder = LabelEncoder()
one_hot_encoder.fit(range(10))


def to_one_hot(values):
    """

    Args:
      values:

    Returns:

    """
    value_idxs = one_hot_encoder.transform([values])
    return torch.eye(len(one_hot_encoder.classes_))[value_idxs]


def log(x):
    """

    Args:
      x:

    Returns:

    """
    return torch.log(x + 1e-8)


def reset_grads(modules: Iterable[nn.Module]):
    """

    Args:
      modules:
    """
    for m in modules:
        m.zero_grad()


def sample_x(X, size):
    """

    Args:
      X:
      size:

    Returns:

    """
    start_idx = numpy.random.randint(0, X.shape[0] - size)
    return X[start_idx : start_idx + size]
