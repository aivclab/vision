from typing import Tuple

__all__ = ["find_extremes"]


def find_extremes(
    c,
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    # determine the most extreme points along the contour

    :param c:
    :type c:
    :return:
    :rtype:
    """

    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    return (extLeft, extTop, extRight, extBot)
