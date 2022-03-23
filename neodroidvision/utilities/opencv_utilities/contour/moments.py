from typing import Tuple, Mapping, Any

import cv2
import numpy
from warg import Number

__all__ = [
    "eccentricity",
    "eccentricity_from_moments",
    "gravity_center",
    "gravity_center_from_moments",
]


def eccentricity_from_moments(moments: Mapping):
    """The eccentricity is calculated by contour moment"""
    a1 = (moments["mu20"] + moments["mu02"]) / 2
    a2 = (
        numpy.sqrt(4 * moments["mu11"] ** 2 + (moments["mu20"] - moments["mu02"]) ** 2)
        / 2
    )
    return numpy.sqrt(1 - (a1 - a2) / (a1 + a2))


def gravity_center_from_moments(moments: Mapping) -> Tuple[Number, Number]:
    """
    # compute the center of the contour

    :return:
    :rtype:
    """
    return (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))


def gravity_center(c: Any) -> Tuple[Number, Number]:
    """
    # compute the center of the contour

    :return:
    :rtype:
    """
    return gravity_center_from_moments(cv2.moments(c))


def eccentricity(c: Any) -> Tuple[Number, Number]:
    """
    # compute the center of the contour

    :return:
    :rtype:
    """
    return eccentricity_from_moments(cv2.moments(c))
