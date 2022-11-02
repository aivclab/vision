#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 5/5/22
           """

import cv2
import numpy
from warg import Number

__all__ = ["dot_points", "is_quadrilateral"]


def dot_points(a: numpy.ndarray, b: numpy.ndarray, c: numpy.ndarray) -> numpy.ndarray:
    """

    :param a:
    :type a: numpy.ndarray
    :param b:
    :type b: numpy.ndarray
    :param c:
    :type c: numpy.ndarray
    :return:
    :rtype: numpy.ndarray
    """
    ab = a - b
    cb = c - b
    return ab / numpy.linalg.norm(ab) @ cb / numpy.linalg.norm(cb)


def is_quadrilateral(contour: numpy.ndarray, threshold: Number = 0.2) -> bool:
    """

    :param contour:
    :type contour:
    :param threshold:
    :type threshold:
    :return:
    :rtype:
    """
    if cv2.isContourConvex(contour):
        for i in range(4):
            if threshold < abs(
                dot_points(contour[i][0], contour[i - 1][0], contour[i - 2][0])
            ):
                return False
        return True
    return False
