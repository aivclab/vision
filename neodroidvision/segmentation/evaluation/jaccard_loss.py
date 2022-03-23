#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"

import numpy
import torch

__all__ = ["jaccard_similarity_score", "jaccard_loss"]


def jaccard_similarity_score(
    pred: torch.Tensor, target: torch.Tensor, *, epsilon: float = 1e-10
) -> torch.Tensor:
    """

    Args:
      pred:
      target:
      epsilon:

    Returns:

    """
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)

    intersection = (pred_flat * target_flat).sum() + epsilon
    union = (target_flat**2).sum() + (pred_flat**2).sum() + epsilon

    dice_coefficient = intersection / (union - intersection)

    return dice_coefficient


def jaccard_loss(
    pred: torch.Tensor, target: torch.Tensor, *, epsilon: float = 1e-10
) -> torch.Tensor:
    """

    Args:
      pred:
      target:
      epsilon:

    Returns:

    """
    return 1 - jaccard_similarity_score(pred, target, epsilon=epsilon)


if __name__ == "__main__":
    h = torch.FloatTensor(numpy.array([[0, 1], [1, 1]]))
    j = torch.FloatTensor(numpy.ones((2, 2)))
    x = jaccard_loss(h, j)
    print(x)
