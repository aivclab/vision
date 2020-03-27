#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 25/03/2020
           """

from itertools import product
from math import sqrt
from typing import Tuple

import torch

from .conversion import iou_of_tensors

__all__ = ["init_prior_box", "assign_priors"]


def init_prior_box(image_size, prior_config):
    """Generate SSD Prior Boxes.
It returns the center, height and width of the priors. The values are relative to the image size
Returns:
  priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the
  values
      are relative to the image size.
"""

    feature_maps = prior_config.FEATURE_MAPS
    min_sizes = prior_config.MIN_SIZES
    max_sizes = prior_config.MAX_SIZES
    strides = prior_config.STRIDES
    aspect_ratios = prior_config.ASPECT_RATIOS
    clip = prior_config.CLIP

    priors = []
    for k, f in enumerate(feature_maps):
        scale = image_size / strides[k]
        for i, j in product(range(f), repeat=2):
            # unit center x,y
            cx = (j + 0.5) / scale
            cy = (i + 0.5) / scale

            # small sized square box
            size = min_sizes[k]
            h = w = size / image_size
            priors.append([cx, cy, w, h])

            # big sized square box
            size = sqrt(min_sizes[k] * max_sizes[k])
            h = w = size / image_size
            priors.append([cx, cy, w, h])

            # change h/w ratio of the small sized box
            size = min_sizes[k]
            h = w = size / image_size
            for ratio in aspect_ratios[k]:
                ratio_sq = sqrt(ratio)
                priors.append([cx, cy, w * ratio_sq, h / ratio_sq])
                priors.append([cx, cy, w / ratio_sq, h * ratio_sq])

    priors_t = torch.tensor(priors)

    if clip:
        priors_t.clamp_(max=1, min=0)

    return priors_t


def assign_priors(
    gt_boxes, gt_labels, corner_form_priors, iou_threshold
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assign ground truth boxes and targets to priors.

Args:
    gt_boxes (num_targets, 4): ground truth boxes.
    gt_labels (num_targets): labels of targets.
    priors (num_priors, 4): corner form priors
Returns:
    boxes (num_priors, 4): real values for priors.
    labels (num_priros): labels for priors.
"""
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
    boxes = gt_boxes[best_target_per_prior_index]
    return boxes, labels
