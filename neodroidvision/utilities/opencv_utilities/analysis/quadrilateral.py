import cv2
import numpy
from warg import Number

__all__ = ["dot_points", "is_quadrilateral"]


def dot_points(a, b, c):
    ab = a - b
    cb = c - b
    return ab / numpy.linalg.norm(ab) @ cb / numpy.linalg.norm(cb)


def is_quadrilateral(contour, threshold: Number = 0.2) -> bool:
    if cv2.isContourConvex(contour):
        for i in range(4):
            if threshold < abs(
                dot_points(contour[i][0], contour[i - 1][0], contour[i - 2][0])
            ):
                return False
        return True
    return False
