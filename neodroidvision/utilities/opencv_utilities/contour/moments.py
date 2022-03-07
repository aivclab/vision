from typing import Tuple

import cv2
import numpy
from warg import Number

__all__ = ["eccentricity_from_moments", "gravity_center"]


def eccentricity_from_moments(moments):
    """The eccentricity is calculated by contour moment"""
    a1 = (moments["mu20"] + moments["mu02"]) / 2
    a2 = (
        numpy.sqrt(4 * moments["mu11"] ** 2 + (moments["mu20"] - moments["mu02"]) ** 2)
        / 2
    )
    return numpy.sqrt(1 - (a1 - a2) / (a1 + a2))


def gravity_center_from_moments(moments) -> Tuple[Number, Number]:
    """
    # compute the center of the contour

    :return:
    :rtype:
    """
    return (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))


def gravity_center(c) -> Tuple[Number, Number]:
    """
    # compute the center of the contour

    :return:
    :rtype:
    """
    return gravity_center_from_moments(cv2.moments(c))
