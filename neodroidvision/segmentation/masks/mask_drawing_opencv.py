#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 20/10/2019
           """

__all__ = ["draw_masks", "ConvexHullEnum", "draw_convex_hull"]

from enum import Enum
from typing import List

import cv2
import numpy
from sorcery import assigned_names


def draw_masks(img2: numpy.ndarray, img_mask_list: List) -> numpy.ndarray:
    """

    Args:
      img2:
      img_mask_list:

    Returns:

    """
    img = img2.copy()
    for ii in range(4):  # for each of the 4 masks
        color_mask = numpy.zeros(img2.shape)
        temp_mask = numpy.ones([img2.shape[0], img2.shape[1]]) * 127.0 / 255.0
        temp_mask[img_mask_list[ii] == 0] = 0
        if ii < 3:  # use different color for each mask
            color_mask[..., ii] = temp_mask
        else:
            (color_mask[..., 0], color_mask[..., 1], color_mask[..., 2]) = (
                temp_mask,
                temp_mask,
                temp_mask,
            )

        img += color_mask
    return img


class ConvexHullEnum(Enum):
    rect, convex, min_area_rect = assigned_names()


def draw_convex_hull(
    mask: numpy.ndarray, mode: ConvexHullEnum = ConvexHullEnum.convex
) -> numpy.ndarray:
    """

    Args:
      mask:
      mode:

    Returns:

    """
    img = numpy.zeros(mask.shape)
    contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if mode == ConvexHullEnum.rect:  # simple rectangle
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
        if mode == ConvexHullEnum.convex:  # minimum convex hull
            hull = cv2.convexHull(c)
            cv2.drawContours(img, [hull], 0, (255, 255, 255), -1)
        elif mode == ConvexHullEnum.min_area_rect:  # minimum area rectangle
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = numpy.int0(box)
            cv2.drawContours(img, [box], 0, (255, 255, 255), -1)
        else:
            raise NotImplementedError
    return img / 255.0
