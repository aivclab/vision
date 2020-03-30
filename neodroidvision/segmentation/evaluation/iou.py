#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 07/03/2020
           """

import numpy

__all__ = ["intersection_over_union"]


def intersection_over_union(img1, img2) -> float:
    img1 = numpy.asarray(img1).astype(numpy.bool)
    img2 = numpy.asarray(img2).astype(numpy.bool)

    intersection = numpy.logical_and(img1, img2)

    return 2.0 * intersection.sum() / (img1.sum() + img2.sum())
