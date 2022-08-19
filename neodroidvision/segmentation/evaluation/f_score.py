#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Callable

import torch

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 20/10/2019
           """

__all__ = ["f_score"]


def f_score(
    pr: torch.Tensor,
    gt: torch.Tensor,
    *,
    beta: float = 1.0,
    eps: float = 1e-7,
    threshold: float = None,
    activation: Callable = torch.sigmoid,
) -> torch.Tensor:
    """

    Args:
    pr (torch.Tensor): A list of predicted elements
    gt (torch.Tensor):  A list of elements that are to be predicted
    eps (float): epsilon to avoid zero division
    threshold: threshold for outputs binarization
    Returns:
    float: IoU (Jaccard) score

    :param pr:
    :param gt:
    :param beta:
    :param eps:
    :param threshold:
    :param activation:
    :return:"""
    if activation:
        pr = activation(pr)

    if threshold is not None:
        pr = (pr > threshold).float()

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta**2) * tp + eps) / (
        (1 + beta**2) * tp + beta**2 * fn + fp + eps
    )

    return score
