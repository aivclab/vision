from typing import Tuple

__all__ = [
    "find_extremes",
    "extent",
    "solidity",
    "pixel_points",
    "point_contour_dist",
    "convexity_defects",
    "equivalent_diameter",
    "orientation",
]

import cv2
import numpy
from warg import Number


def find_extremes(
    c: numpy.ndarray,
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    # determine the most extreme points along the contour

    :param c:
    :type c:
    :return:
    :rtype:
    """

    ext_left = tuple(c[c[:, :, 0].argmin()][0])
    ext_right = tuple(c[c[:, :, 0].argmax()][0])
    ext_top = tuple(c[c[:, :, 1].argmin()][0])
    ext_bot = tuple(c[c[:, :, 1].argmax()][0])

    return (ext_left, ext_top, ext_right, ext_bot)


def extent(cnt: numpy.ndarray) -> Number:
    """Extent is the ratio of contour area to bounding rectangle area."""
    x, y, w, h = cv2.boundingRect(cnt)
    return cv2.contourArea(cnt) / (w * h)


def solidity(cnt: numpy.ndarray) -> Number:
    """
    Solidity is the ratio of contour area to its convex hull area.


    :return:
    :rtype:
    """

    return cv2.contourArea(cnt) / cv2.contourArea(cv2.convexHull(cnt))


def point_contour_dist(cnt: numpy.ndarray, pnt: Tuple[int, int] = (50, 50)) -> Number:
    """
    In  the  function, third argument is " measureDist".
    If it is True, it finds the signed distance.
    If False, it finds only if the point is inside or outside or on the contour.

    :return:
    :rtype:




    """
    return cv2.pointPolygonTest(cnt, pnt, measureDist=True)


def equivalent_diameter(cnt: numpy.ndarray) -> Number:
    """Equivalent Diameter is the diameter of the circle whose area is same as the contour area."""

    return numpy.sqrt(4 * cv2.contourArea(cnt) / numpy.pi)


def orientation(cnt: numpy.ndarray) -> Number:
    """Orientation is the angle at which object is directed."""

    (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
    return angle


def pixel_points(im_shape, cnt: numpy.ndarray) -> numpy.ndarray:
    """
    In some cases, we may need all the points which comprises that object. It can be done as follows:


      :param cnt:
      :type cnt:
      :return:
      :rtype:
    """
    mask = numpy.zeros(im_shape, numpy.uint8)
    cv2.drawContours(mask, [cnt], 0, 255, -1)
    return numpy.transpose(numpy.nonzero(mask))


def convexity_defects(cnt: numpy.ndarray) -> numpy.ndarray:
    """
    Notice that "returnPoints = False" in first line to get indices of the contour points, because input to convexityDefects() should be these indices, not original points.

    It returns a defects structure, an array of four values - [ start point, end point, farthest point, approximate distance to farthest point ]
    """
    return cv2.convexityDefects(cnt, cv2.convexHull(cnt, returnPoints=False))


if __name__ == "__main__":
    from neodroidvision.utilities.misc.perlin import generate_perlin_noise
    from draugr.opencv_utilities import threshold_channel, show_image

    a = generate_perlin_noise() * 255
    a = numpy.uint8(a)
    t = threshold_channel(a)
    show_image(t, wait=True)
