#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 30/06/2020
           """

import torch
from torch.nn import functional

__all__ = ["PairwiseRankingLoss"]


class PairwiseRankingLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Neighbours(same category) are pulled together and non-neighbors are pushed apart

    From http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf"""

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self._margin = margin

    def forward(
        self, anchor: torch.Tensor, other: torch.Tensor, is_diff: torch.Tensor
    ) -> torch.Tensor:
        """

        if the is_diff is 0 the examples are of the same category and thus gradient point is the direction to
        minimize distance between the examples.
        if the is_diff is 1 it should minimize the residual of margin-distance to spread samples provided apart
        in the latent space.

        # Reduction is mean

        :param anchor:
        :type anchor:
        :param other:
        :type other:
        :param is_diff:
        :type is_diff:
        :return:
        :rtype:"""
        # assert s1.is_contiguous()
        # assert s2.is_contiguous()
        # assert is_same.is_contiguous()

        euclidean_distance = functional.pairwise_distance(anchor, other)
        return torch.mean(
            (1 - is_diff) * euclidean_distance**2
            + is_diff * torch.clamp(self._margin - euclidean_distance, min=0.0) ** 2
        )  # if distance is larger than margin(desirable), clip to 0 loss.
