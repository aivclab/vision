#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 5/5/22
           """

from pathlib import Path
from typing import Any

import cv2
from draugr.opencv_utilities import show_image

__all__ = ["detect_lines"]


def detect_lines(img: Any, debug: bool = True) -> Any:
    """

    :param img:
    :type img:
    :param debug:
    :type debug:
    :return:
    :rtype:
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(img)[0]

    if debug:
        drawn_img = lsd.drawSegments(img, lines)
        show_image(drawn_img, wait=True)

    return lines


if __name__ == "__main__":
    file = "white-paper-table-13912022.jpg"
    # file = "2.jpg"
    # file = "NdNLO.jpg"

    i = cv2.imread(str(Path.home() / "Pictures" / file))
    detect_lines(i)
