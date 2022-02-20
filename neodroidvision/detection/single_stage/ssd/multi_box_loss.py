#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/03/2020
           """

from typing import Tuple

import torch
from draugr.torch_utilities.operations.enums import ReductionMethodEnum
from torch import nn
from torch.nn import functional
from warg import Number

from neodroidvision.detection.single_stage.ssd.bounding_boxes import (
    hard_negative_mining,
)

__all__ = ["MultiBoxLoss"]


class MultiBoxLoss(nn.Module):
    """ """

    def __init__(self, neg_pos_ratio: Number):
        """Implement SSD MultiBox Loss.

        Basically, MultiBox loss combines classification loss
        and Smooth L1 regression loss."""
        super().__init__()
        self._neg_pos_ratio = neg_pos_ratio

    def forward(
        self,
        confidence: torch.Tensor,
        predicted_locations: torch.Tensor,
        labels: torch.Tensor,
        gt_locations: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute classification loss and smooth l1 loss.

        Args:
        confidence (batch_size, num_priors, num_categories): class predictions.
        predicted_locations (batch_size, num_priors, 4): predicted locations.
        labels (batch_size, num_priors): real labels of all the priors.
        gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors."""

        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            mask = hard_negative_mining(
                -functional.log_softmax(confidence, dim=2)[:, :, 0],
                labels,
                self._neg_pos_ratio,
            )

        pos_mask = labels > 0
        gt_locations_masked = gt_locations[pos_mask, :].reshape(-1, 4)
        num_pos = gt_locations_masked.size(0)

        return (
            functional.smooth_l1_loss(
                predicted_locations[pos_mask, :].reshape(-1, 4),
                gt_locations_masked,
                reduction=ReductionMethodEnum.sum.value,
            )
            / num_pos,
            functional.cross_entropy(
                confidence[mask, :].reshape(-1, confidence.size(2)),
                labels[mask],
                reduction=ReductionMethodEnum.sum.value,
            )
            / num_pos,
        )
