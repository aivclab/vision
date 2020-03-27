#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 25/03/2020
           """

from typing import Tuple

import numpy
import torch

from draugr.opencv_utilities import (
    CV2PhotometricDistort,
    ConvertFromInts,
    CV2Expand,
    CV2RandomSampleCrop,
    CV2RandomMirror,
    CV2ToPercentCoords,
    CV2Resize,
    SubtractMeans,
    CV2ToTensor,
    CV2Compose,
)
from neodroidvision.data.datasets.supervised.splitting import Split
from .conversion import (
    center_form_to_corner_form,
    convert_boxes_to_locations,
    corner_form_to_center_form,
)
from neodroidvision.detection.single_stage.ssd.bounding_boxes.ssd_priors import (
    assign_priors,
    init_prior_box,
)

__all__ = ["SSDTransform", "SSDTargetTransform"]


class SSDTransform(torch.nn.Module):
    def __init__(self, IMAGE_SIZE, PIXEL_MEAN, split: Split):
        super().__init__()

        transform = []

        if split == Split.Training:
            transform.extend(
                [
                    ConvertFromInts(),
                    CV2PhotometricDistort(),
                    CV2Expand(PIXEL_MEAN),
                    CV2RandomSampleCrop(),
                    CV2RandomMirror(),
                    CV2ToPercentCoords(),
                ]
            )

        transform.extend(
            [CV2Resize(IMAGE_SIZE), SubtractMeans(PIXEL_MEAN), CV2ToTensor()]
        )

        self.transforms = CV2Compose(transform)

    def __call__(self, x):
        return self.transforms(x)


class SSDTargetTransform(torch.nn.Module):
    def __init__(
        self, IMAGE_SIZE, PRIORS, center_variance, size_variance, iou_threshold
    ):
        super().__init__()
        self.center_form_priors = init_prior_box(IMAGE_SIZE, PRIORS)
        self.corner_form_priors = center_form_to_corner_form(self.center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels) -> Tuple:
        if type(gt_boxes) is numpy.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)

        if type(gt_labels) is numpy.ndarray:
            gt_labels = torch.from_numpy(gt_labels)

        boxes, labels = assign_priors(
            gt_boxes, gt_labels, self.corner_form_priors, self.iou_threshold
        )
        locations = convert_boxes_to_locations(
            corner_form_to_center_form(boxes),
            self.center_form_priors,
            self.center_variance,
            self.size_variance,
        )

        return locations, labels
