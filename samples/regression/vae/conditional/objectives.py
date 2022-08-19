#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Christian Heider Nielsen"
__doc__ = r"""
           """

import torch
from draugr.torch_utilities import ReductionMethodEnum
from torch.nn.functional import binary_cross_entropy


def loss_fn(recon_x, x, mean, log_var):
    """

    :param recon_x:
    :type recon_x:
    :param x:
    :type x:
    :param mean:
    :type mean:
    :param log_var:
    :type log_var:
    :return:
    :rtype:"""
    bce = binary_cross_entropy(
        recon_x.view(-1, 28 * 28),
        x.view(-1, 28 * 28),
        reduction=ReductionMethodEnum.sum.value,
    )
    kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return (bce + kld) / x.size(0)
