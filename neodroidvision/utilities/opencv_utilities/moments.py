from typing import Tuple

import cv2
import numpy
from warg import Number

__all__ = ["eccentricity_from_moments", "contour_centroid"]


def eccentricity_from_moments(moments):
    """The eccentricity is calculated by contour moment"""
    a1 = (moments["mu20"] + moments["mu02"]) / 2
    a2 = (
        numpy.sqrt(4 * moments["mu11"] ** 2 + (moments["mu20"] - moments["mu02"]) ** 2)
        / 2
    )
    return numpy.sqrt(1 - (a1 - a2) / (a1 + a2))


def contour_centroid(c) -> Tuple[Number, Number]:
    """
    # compute the center of the contour

    :return:
    :rtype:
    """
    moments = cv2.moments(c)

    centroid_x = int(moments["m10"] / moments["m00"])
    centroid_y = int(moments["m01"] / moments["m00"])

    return (centroid_x, centroid_y)
