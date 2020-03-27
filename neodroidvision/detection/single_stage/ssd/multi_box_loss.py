#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/03/2020
           """

from typing import Tuple

import torch
from torch import nn as nn
from torch.nn import functional as F

from neodroidvision.detection.single_stage.ssd.bounding_boxes import conversion

__all__ = ["MultiBoxLoss"]


class MultiBoxLoss(nn.Module):
    def __init__(self, neg_pos_ratio: float):
        """Implement SSD MultiBox Loss.

Basically, MultiBox loss combines classification loss
 and Smooth L1 regression loss.
"""
        super().__init__()
        self._neg_pos_ratio = neg_pos_ratio

    def forward(
        self, confidence, predicted_locations, labels, gt_locations
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute classification loss and smooth l1 loss.

Args:
    confidence (batch_size, num_priors, num_classes): class predictions.
    predicted_locations (batch_size, num_priors, 4): predicted locations.
    labels (batch_size, num_priors): real labels of all the priors.
    gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
"""
        num_classes = confidence.size(2)

        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = conversion.hard_negative_mining(loss, labels, self._neg_pos_ratio)

        confidence_masked = confidence[mask, :]
        classification_loss = F.cross_entropy(
            confidence_masked.view(-1, num_classes), labels[mask], reduction="sum"
        )

        pos_mask = labels > 0
        predicted_locations_masked = predicted_locations[pos_mask, :].view(-1, 4)
        gt_locations_masked = gt_locations[pos_mask, :].view(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(
            predicted_locations_masked, gt_locations_masked, reduction="sum"
        )

        num_pos = gt_locations_masked.size(0)
        return smooth_l1_loss / num_pos, classification_loss / num_pos
