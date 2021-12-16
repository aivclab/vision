#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 11/05/2020
           """

import math

import torch

__all__ = ["area_of_tensors", "iou_of_tensors", "hard_negative_mining"]


def area_of_tensors(left_top: torch.Tensor, right_bottom: torch.Tensor) -> torch.Tensor:
    """Compute the areas of rectangles given two corners.

    Args:
    left_top (N, 2): left top corner.
    right_bottom (N, 2): right bottom corner.

    Returns:
    area (N): return the area."""
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def iou_of_tensors(
    boxes0: torch.Tensor, boxes1: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
    boxes0 (N, 4): ground truth boxes.
    boxes1 (N or 1, 4): predicted boxes.
    eps: a small number to avoid 0 as denominator.
    Returns:
    iou (N): IoU values."""
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of_tensors(overlap_left_top, overlap_right_bottom)
    area0 = area_of_tensors(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of_tensors(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_negative_mining(
    loss: torch.Tensor, labels: torch.Tensor, neg_pos_ratio: float
) -> torch.Tensor:
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
    cut the number of negative predictions to make sure the ratio
    between the negative examples and positive examples is no more
    the given ratio for an image.

    Args:
    loss (N, num_priors): the loss for each example.
    labels (N, num_priors): the labels.
    neg_pos_ratio:  the ratio between the negative examples and positive examples."""
    pos_mask = labels > 0

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < pos_mask.long().sum(dim=1, keepdim=True) * neg_pos_ratio
    return pos_mask | neg_mask
