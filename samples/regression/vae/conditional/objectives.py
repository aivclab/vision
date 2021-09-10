#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
           """


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
    bce = torch.nn.functional.binary_cross_entropy(
        recon_x.view(-1, 28 * 28), x.view(-1, 28 * 28), reduction="sum"
    )
    kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return (bce + kld) / x.size(0)
