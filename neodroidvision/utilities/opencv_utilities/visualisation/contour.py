import cv2
import numpy

from neodroidvision.utilities.opencv_utilities.contour.analysis import convexity_defects


def draw_convexity_defects(img: numpy.ndarray, cnt: numpy.ndarray) -> numpy.ndarray:
    """

    :return:
    :rtype:
    """
    defects = convexity_defects(cnt)

    for i in range(defects.shape[0]):
        start, end, far, d = defects[i, 0]
        cv2.line(img, tuple(cnt[start][0]), tuple(cnt[end][0]), [0, 255, 0], 2)
        cv2.circle(img, tuple(cnt[far][0]), 5, [0, 0, 255], -1)

    return img
