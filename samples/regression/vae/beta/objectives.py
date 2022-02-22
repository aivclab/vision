#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numbers import Number

import torch
from torch.nn.functional import (
    mse_loss,
    binary_cross_entropy_with_logits,
    binary_cross_entropy,
)

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
Objective functions for beta vae's
           """


def loss_function(reconstruction, original, mean, log_var, beta: Number = 1):
    """

    Args:
      reconstruction:
      original:
      mean:
      log_var:
      beta:

    Returns:

    """
    return reconstruction_loss(reconstruction, original) + beta * kl_divergence(
        mean, log_var
    )


def reconstruction_loss(reconstruction, original):
    """

    Args:
      reconstruction:
      original:

    Returns:

    """
    batch_size = original.size(0)
    assert batch_size != 0

    reconstruction = torch.sigmoid(reconstruction)

    # return binary_cross_entropy(reconstruction,    original,    size_average=False).div(batch_size)
    return mse_loss(reconstruction, original, size_average=False).div(batch_size)


def kl_divergence(mean, log_var):
    """

    Args:
      mean:
      log_var:

    Returns:

    """
    batch_size = mean.size(0)
    assert batch_size != 0

    if mean.data.ndimension() == 4:
        mean = mean.view(mean.size(0), mean.size(1))

    if log_var.data.ndimension() == 4:
        log_var = log_var.view(log_var.size(0), log_var.size(1))

    klds = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
    total_kld = klds.sum(1).mean(0, True)
    # dimension_wise_kld = klds.mean(0)
    # mean_kld = klds.mean(1).mean(0, True)

    return total_kld  # , dimension_wise_kld, mean_kld
