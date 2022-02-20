#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 11/05/2020
           """

from torch import nn

from neodroidvision.detection.single_stage.ssd.architecture.nms_box_heads.box_predictor import (
    BoxPredictor,
)
from neodroidvision.utilities import SeparableConv2d

__all__ = ["SSDLiteBoxPredictor"]


class SSDLiteBoxPredictor(BoxPredictor):
    """ """

    def category_block(
        self, level: int, out_channels: int, boxes_per_location: int
    ) -> nn.Module:
        """

        :param level:
        :type level:
        :param out_channels:
        :type out_channels:
        :param boxes_per_location:
        :type boxes_per_location:
        :return:
        :rtype:"""
        if level == len(self.out_channels) - 1:
            return nn.Conv2d(
                out_channels, boxes_per_location * self.num_categories, kernel_size=1
            )

        return SeparableConv2d(
            out_channels,
            boxes_per_location * self.num_categories,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def location_block(
        self, level: int, out_channels: int, boxes_per_location: int
    ) -> nn.Module:
        """

        :param level:
        :type level:
        :param out_channels:
        :type out_channels:
        :param boxes_per_location:
        :type boxes_per_location:
        :return:
        :rtype:"""
        if level == len(self.out_channels) - 1:
            return nn.Conv2d(out_channels, boxes_per_location * 4, kernel_size=1)
        return SeparableConv2d(
            out_channels, boxes_per_location * 4, kernel_size=3, stride=1, padding=1
        )
