#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 04/03/2020
           """

from neodroidvision.detection.single_stage.ssd.architecture.anchors.prior_box import (
    PriorBox,
)
from neodroidvision.detection.single_stage.ssd.ssd_utilities import (
    Compose,
    ConvertFromInts,
    Expand,
    PhotometricDistort,
    RandomMirror,
    RandomSampleCrop,
    Resize,
    SSDTargetTransform,
    SubtractMeans,
    ToPercentCoords,
    ToTensor,
)

__all__ = ["build_target_transform", "build_transforms"]


def build_transforms(cfg, is_train=True):
    if is_train:
        transform = [
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(cfg.INPUT.PIXEL_MEAN),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
        ]
    else:
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
        ]
    transform = Compose(transform)
    return transform


def build_target_transform(cfg):
    transform = SSDTargetTransform(
        PriorBox(cfg)(),
        cfg.MODEL.CENTER_VARIANCE,
        cfg.MODEL.SIZE_VARIANCE,
        cfg.MODEL.THRESHOLD,
    )
    return transform
