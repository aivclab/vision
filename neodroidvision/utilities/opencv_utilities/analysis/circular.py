import math
from typing import Tuple, List

import cv2
import numpy


def daugman(
    gray_img: numpy.ndarray,
    center: Tuple[int, int],
    start_r: int,
    end_r: int,
    step: int = 1,
) -> Tuple[float, int]:
    """The function will calculate pixel intensities for the circles
    in the ``range(start_r, end_r, step)`` for a given ``center``,
    and find a circle that precedes the biggest intensity drop

    :param gray_img: grayscale picture
    :param center:  center coordinates ``(x, y)``
    :param start_r: bottom value for iris radius in pixels
    :param end_r: top value for iris radius in pixels
    :param step: step value for iris radii range in pixels

    .. attention::
        Input grayscale image should be a square, not a rectangle

    :return: intensity_value, radius
    """
    # x, y = center
    intensities = []
    mask = numpy.zeros_like(gray_img)

    radii = list(
        range(start_r, end_r, step)
    )  # type: List[int]     # for every radius in range
    for r in radii:
        cv2.circle(mask, center, r, 255, 1)  # draw circle on mask

        diff = (
            gray_img & mask
        )  # get pixel from original image, it is faster than np or cv2

        intensities.append(
            numpy.add.reduce(diff[diff > 0]) / (2 * math.pi * r)
        )  # normalize, numpy.add.reduce faster than .sum()
        #            diff[diff > 0] faster than .flatten()

        mask.fill(0)  # refresh mask

    intensities_np = numpy.array(
        intensities, dtype=numpy.float32
    )  # calculate delta of radius intensitiveness     #     mypy does not tolerate var type reload
    del intensities

    intensities_np = (
        intensities_np[:-1] - intensities_np[1:]
    )  # circles intensity differences, x5 faster than numpy.diff()

    intensities_np = abs(cv2.GaussianBlur(intensities_np, (1, 5), 0))
    # apply gaussian filter
    #     GaussianBlur() faster than filter2D() with custom kernel
    # original kernel:
    # > The Gaussian filter in our case is designedin MATLAB and
    # > is a 1 by 5 (rows by columns) vector with intensity values
    # > given by vector A = [0.0003 0.1065 0.7866 0.1065 0.0003]

    idx = numpy.argmax(intensities_np)  # type: int     # get maximum value

    return intensities_np[idx], radii[idx]


if __name__ == "__main__":
    pass
    daugman()
