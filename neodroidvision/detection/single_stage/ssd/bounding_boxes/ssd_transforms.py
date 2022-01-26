#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 25/03/2020
           """

from typing import Any, Tuple

import numpy
import torch
from draugr.numpy_utilities import SplitEnum
from draugr.opencv_utilities import (
    CV2Compose,
    CV2Expand,
    CV2PhotometricDistort,
    CV2RandomMirror,
    CV2RandomSampleCrop,
    CV2Resize,
    CV2ToPercentCoords,
    CV2ToTensor,
    ConvertFromInts,
    SubtractMeans,
)
from warg import NOD

from neodroidvision.detection.single_stage.ssd.bounding_boxes.ssd_priors import (
    build_priors,
    ssd_assign_priors,
)
from .conversion import (
    center_to_corner_form,
    convert_boxes_to_locations,
    corner_form_to_center_form,
)

__all__ = ["SSDTransform", "SSDAnnotationTransform"]


class SSDTransform(torch.nn.Module):
    """ """

    def __init__(self, image_size: Tuple, pixel_mean: Tuple, split: SplitEnum):
        """

        :param image_size:
        :type image_size:
        :param pixel_mean:
        :type pixel_mean:
        :param split:
        :type split:"""
        super().__init__()

        transform_list = []

        if split == SplitEnum.training:
            transform_list.extend(
                [
                    ConvertFromInts(),
                    CV2PhotometricDistort(),
                    CV2Expand(pixel_mean),
                    CV2RandomSampleCrop(),
                    CV2RandomMirror(),
                    CV2ToPercentCoords(),
                ]
            )

        transform_list.extend(
            [CV2Resize(image_size), SubtractMeans(pixel_mean), CV2ToTensor()]
        )

        self.transforms = CV2Compose(transform_list)

    def __call__(self, *x) -> torch.Tensor:
        """

        :param x:
        :type x:
        :return:
        :rtype:"""
        return self.transforms(*x)


class SSDAnnotationTransform(torch.nn.Module):
    """ """

    def __init__(
        self,
        *,
        image_size: Any,
        priors_cfg: NOD,
        center_variance: Any,
        size_variance: Any,
        iou_threshold: Any
    ):
        """

        :param image_size:
        :type image_size:
        :param priors_cfg:
        :type priors_cfg:
        :param center_variance:
        :type center_variance:
        :param size_variance:
        :type size_variance:
        :param iou_threshold:
        :type iou_threshold:"""
        super().__init__()
        self.center_form_priors = build_priors(image_size=image_size, **priors_cfg)
        self.corner_form_priors = center_to_corner_form(self.center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes: Any, gt_labels: Any) -> Tuple:
        """

        :param gt_boxes:
        :type gt_boxes:
        :param gt_labels:
        :type gt_labels:
        :return:
        :rtype:"""
        if type(gt_boxes) is numpy.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)

        if type(gt_labels) is numpy.ndarray:
            gt_labels = torch.from_numpy(gt_labels)

        boxes, labels = ssd_assign_priors(
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            corner_form_priors=self.corner_form_priors,
            iou_threshold=self.iou_threshold,
        )
        locations = convert_boxes_to_locations(
            center_form_boxes=corner_form_to_center_form(boxes),
            center_form_priors=self.center_form_priors,
            center_variance=self.center_variance,
            size_variance=self.size_variance,
        )

        return locations, labels
