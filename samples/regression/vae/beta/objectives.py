#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numbers import Number

import torch
from draugr.writers import Writer, MockWriter
from torch.nn.functional import (
    mse_loss,
)

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
Objective functions for beta vae's
           """


def loss_function(
    reconstruction,
    original,
    mean,
    log_var,
    beta: Number = 1,
    writer: Writer = MockWriter(),
):
    """

    Args:
      reconstruction:
      original:
      mean:
      log_var:
      beta:

    Returns:

    """

    total_kld = kl_divergence(mean, log_var, writer)

    if True:
        beta_vae_loss = beta * total_kld
    else:
        beta_vae_loss = (
            recon_loss
            + self.gamma
            * (
                total_kld
                - torch.clamp(
                    self.C_max / self.C_stop_iter * self.global_iter,
                    0,
                    self.C_max.data[0],
                )  # C
            ).abs()
        )

    return reconstruction_loss(reconstruction, original) + beta_vae_loss


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


def kl_divergence(mean, log_var, writer: Writer = MockWriter()) -> torch.Tensor:
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

    if writer:
        writer.scalar("dimension_wise_kld", klds.mean(0))
        writer.scalar("mean_kld", klds.mean(1).mean(0, True))

    return klds.sum(1).mean(0, True)  # total_kld
