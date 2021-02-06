#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 25/03/2020
           """

from itertools import product
from math import sqrt
from typing import Tuple

import numpy
import torch
from warg import drop_unused_kws

from .tensor_metrics import iou_of_tensors

__all__ = ["build_priors", "ssd_assign_priors"]


@drop_unused_kws
def build_priors(
    *,
    image_size: numpy.ndarray,
    feature_maps: torch.Tensor,
    min_sizes: torch.Tensor,
    max_sizes: torch.Tensor,
    strides: torch.Tensor,
    aspect_ratios: torch.Tensor,
    clip: bool = True
) -> torch.Tensor:
    """Generate SSD Prior Boxes.
    It returns the center, height and width of the priors. The values are relative to the image size
    Returns:
    priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the
    values
    are relative to the image size."""

    priors = []
    for k, f in enumerate(feature_maps):
        scale = image_size / strides[k]
        small_hw = min_sizes[k] / image_size
        big_hw = sqrt(min_sizes[k] * max_sizes[k]) / image_size

        for i, j in product(range(f), repeat=2):  # all pair combinations
            # unit center x,y
            cx = (j + 0.5) / scale
            cy = (i + 0.5) / scale

            # small sized square box
            h = w = small_hw
            priors.append([cx, cy, w, h])

            # big sized square box
            h = w = big_hw
            priors.append([cx, cy, w, h])

            # change h/w ratio of the small sized box
            h = w = small_hw
            for ratio in aspect_ratios[k]:
                ratio_sq = sqrt(ratio)
                priors.append([cx, cy, w * ratio_sq, h / ratio_sq])
                priors.append([cx, cy, w / ratio_sq, h * ratio_sq])

    priors_t = torch.tensor(priors)

    if clip:
        priors_t.clamp_(min=0, max=1)

    return priors_t


def ssd_assign_priors(
    *,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    corner_form_priors: torch.Tensor,
    iou_threshold: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assign ground truth boxes and targets to priors.

    Args:
    gt_boxes (num_targets, 4): ground truth boxes.
    gt_labels (num_targets): labels of targets.
    priors (num_priors, 4): corner form priors
    Returns:
    boxes (num_priors, 4): real values for priors.
    labels (num_priros): labels for priors.
    :param gt_boxes:
    :type gt_boxes:
    :param gt_labels:
    :type gt_labels:
    :param corner_form_priors:
    :type corner_form_priors:
    :param iou_threshold:
    :type iou_threshold:"""
    # size: num_priors x num_targets
    ious = iou_of_tensors(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))
    # size: num_priors
    best_target_per_prior, best_target_per_prior_index = ious.max(1)
    # size: num_targets
    best_prior_per_target, best_prior_per_target_index = ious.max(0)

    for target_index, prior_index in enumerate(best_prior_per_target_index):
        best_target_per_prior_index[prior_index] = target_index
    # 2.0 is used to make sure every target has a prior assigned
    best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
    # size: num_priors
    labels = gt_labels[best_target_per_prior_index]
    labels[best_target_per_prior < iou_threshold] = 0  # the backgournd id
    return gt_boxes[best_target_per_prior_index], labels
