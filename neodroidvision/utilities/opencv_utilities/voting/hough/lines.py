from typing import Any

import cv2
import numpy
from draugr.opencv_utilities import to_gray, show_image, ThresholdTypeFlag


def hough_lines(
    img,
    kernel_size=11,
    sigma=1.4,  # 0
    aperture_size=3,
    rho=1,
    theta=numpy.pi / 180,
    min_votes=99,
    lines=100,
    min_line_length=10,
    max_line_gap=250,
    debug: bool = False,
) -> Any:
    gray = to_gray(img)

    if True:  # remove noise
        # gray = cv2.medianBlur(gray, kernel_size)
        gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)

    if False:
        if False:
            edges = cv2.Canny(
                gray, threshold1=50, threshold2=200, apertureSize=aperture_size
            )
        else:
            adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            thresh_type = cv2.THRESH_BINARY_INV
            edges = cv2.adaptiveThreshold(gray, 255, adapt_type, thresh_type, 11, 2)
    else:
        laplacian = cv2.Laplacian(
            gray, cv2.CV_8UC1, ksize=3  # ,cv2.CV_16UC1, #cv2.CV_16S, # cv2.CV_64F
        )
        # blurryness = resLap.var()
        # sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
        # sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)

        edges = cv2.threshold(
            laplacian,
            0,
            255,
            ThresholdTypeFlag.otsu.value
            + ThresholdTypeFlag.binary.value,  # ThresholdTypeFlag.to_zero.value
        )[1]

    if True:
        lines = cv2.HoughLinesP(
            edges,
            rho=rho,
            theta=theta,
            threshold=min_votes,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )
    else:
        lines = None

    if debug:
        show_image(gray)
        # show_image(laplacian)
        show_image(edges, wait=True)
        if False:
            for line in lines:  # Draw lines on the image
                x1, y1, x2, y2 = line[0]
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

    return lines
