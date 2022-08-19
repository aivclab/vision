#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 07/03/2020
           """

import numpy

__all__ = ["intersection_over_union", "dice_coefficient"]


def dice_coefficient(img1, img2, threshold=0.5) -> float:
    """

    Args:
      img1:
      img2:

    Returns:
    :param threshold:
    :type threshold:
    """
    img1 = numpy.asarray(img1) > threshold
    img2 = numpy.asarray(img2) > threshold

    intersection = numpy.logical_and(img1, img2)
    return 2.0 * intersection.sum() / (img1.sum() + img2.sum())


def intersection_over_union(img1, img2, threshold=0.5) -> float:
    """

    :param img1:
    :type img1:
    :param img2:
    :type img2:
    :param threshold:
    :type threshold:
    :return:
    :rtype:
    """
    img1 = numpy.asarray(img1) > threshold
    img2 = numpy.asarray(img2) > threshold

    return numpy.sum(numpy.logical_and(img1, img2)) / numpy.sum(
        numpy.logical_or(img1, img2)
    )
