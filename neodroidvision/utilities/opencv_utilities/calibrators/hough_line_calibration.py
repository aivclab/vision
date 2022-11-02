#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 5/5/22
           """

from math import pi
from pathlib import Path

import cv2
import numpy
from draugr.opencv_utilities import add_trackbar, match_return_code, show_image

__all__ = ["hough_line_calibrator"]


def hough_line_calibrator(
    frames: numpy.ndarray,
    float_multiplier=1000,
) -> None:  # TODO: GENERALISE INTERACTIVE CALIBRATOR TO MANY MORE OPENCV FUNCTIONS
    """description"""
    a_key_pressed = None  # .init
    edges = None

    lo_label = "Lo_Threshold"
    lo = 60
    lo_prev = -1

    hi_label = "Hi_Threshold"
    hi = 80
    hi_prev = -1

    rho_label = f"Rho_*{float_multiplier}"
    rho = int(1 * float_multiplier)
    rho_prev = -1

    theta_label = f"Theta_*{float_multiplier}"
    theta = int((pi / 180) * float_multiplier)
    theta_prev = -1

    threshold_label = "threshold"
    threshold = 100
    threshold_prev = -1

    srn_label = "srn"
    srn = 50
    srn_prev = -1

    stn_label = "stn"
    stn = 10
    stn_prev = -1

    min_theta_label = f"min_theta_*{float_multiplier}"
    min_theta = 0
    min_theta_prev = -1

    max_theta_label = f"max_theta_*{float_multiplier}"
    max_theta = int((pi) * float_multiplier)
    max_theta_prev = -1

    frame_window_label = "Frame"
    canny_frame_window_label = f"{frame_window_label}.Canny"
    canny_hough_lines_window_label = f"{canny_frame_window_label}.Lines"

    frame_generator = iter(frames)
    frame = next(frame_generator)
    show_image(frame, frame_window_label)

    show_image(frame, canny_frame_window_label)

    add_trackbar(canny_frame_window_label, lo_label, default=lo, max_val=1000)
    add_trackbar(canny_frame_window_label, hi_label, default=hi, max_val=1000)

    show_image(frame, canny_hough_lines_window_label)

    add_trackbar(
        canny_hough_lines_window_label,
        rho_label,
        default=rho,
        min_val=1 * float_multiplier,
        max_val=10 * float_multiplier,
    )
    add_trackbar(
        canny_hough_lines_window_label,
        theta_label,
        default=theta,
        min_val=int((pi / 180) * float_multiplier),
        max_val=pi * float_multiplier,
    )
    add_trackbar(
        canny_hough_lines_window_label, threshold_label, default=threshold, max_val=1000
    )
    add_trackbar(
        canny_hough_lines_window_label,
        srn_label,
        default=srn,
    )
    add_trackbar(
        canny_hough_lines_window_label,
        stn_label,
        default=stn,
    )
    add_trackbar(
        canny_hough_lines_window_label,
        min_theta_label,
        default=min_theta * float_multiplier,
        max_val=pi * float_multiplier,
    )
    add_trackbar(
        canny_hough_lines_window_label,
        max_theta_label,
        default=max_theta * float_multiplier,
        max_val=pi * float_multiplier,
    )

    print(f"{'press n for next image, press [ESC] or q to exit ':-^60}")
    new_image = False
    while True:
        if match_return_code(a_key_pressed, "n"):
            frame = next(frame_generator)
            cv2.imshow(frame_window_label, frame)
            new_image = True

        if match_return_code(a_key_pressed):  # quit
            break

        lo = cv2.getTrackbarPos(lo_label, canny_frame_window_label)
        hi = cv2.getTrackbarPos(hi_label, canny_frame_window_label)

        if (
            lo != lo_prev or hi != hi_prev or new_image
        ):  # --------------------------= RE-SYNC
            new_image = False
            a_canny_refresh_flag = True  # --------------------------= FLAG

            lo_prev = lo
            hi_prev = hi
        else:

            a_canny_refresh_flag = False  # --------------------------= Un-FLAG

        threshold = cv2.getTrackbarPos(threshold_label, canny_hough_lines_window_label)
        srn = cv2.getTrackbarPos(srn_label, canny_hough_lines_window_label)
        stn = cv2.getTrackbarPos(stn_label, canny_hough_lines_window_label)
        min_theta = float(
            cv2.getTrackbarPos(min_theta_label, canny_hough_lines_window_label)
            / float_multiplier
        )
        max_theta = float(
            cv2.getTrackbarPos(max_theta_label, canny_hough_lines_window_label)
            / float_multiplier
        )
        rho = float(
            cv2.getTrackbarPos(rho_label, canny_hough_lines_window_label)
            / float_multiplier
        )
        theta = float(
            cv2.getTrackbarPos(theta_label, canny_hough_lines_window_label)
            / float_multiplier
        )

        if (
            rho != rho_prev
            or theta != theta_prev
            or threshold != threshold_prev
            or srn != srn_prev
            or stn != stn_prev
            or min_theta != min_theta_prev
            or max_theta != max_theta_prev
        ):  # ----------------------------------------------= RE-SYNC

            a_hough_refresh_flag = True  # --------------------------= FLAG

            rho_prev = rho
            theta_prev = theta
            threshold_prev = threshold
            srn_prev = srn
            stn_prev = stn
            min_theta_prev = min_theta
            max_theta_prev = max_theta
        else:

            a_hough_refresh_flag = False  # --------------------------= Un-FLAG

        if (
            a_canny_refresh_flag
        ):  # REFRESH-process-pipe-line ( with recent <state> <vars> )
            edges = cv2.Canny(frame, lo, hi)

            cv2.imshow(canny_frame_window_label, edges)

        if a_canny_refresh_flag or a_hough_refresh_flag:
            lines = cv2.HoughLines(
                edges,
                rho=rho,
                theta=theta,
                threshold=threshold,
                srn=srn,
                stn=stn,
                min_theta=min_theta,
                max_theta=max_theta,
            )

            demo_with_lines = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # .re-init <<< src
            demo_with_lines = cv2.cvtColor(demo_with_lines, cv2.COLOR_RGB2BGR)
            if lines is not None:
                draw_lines(demo_with_lines, lines)

            cv2.imshow(canny_hough_lines_window_label, demo_with_lines)

        a_key_pressed = cv2.waitKey(1) & 0xFF

    cv2.destroyWindow(frame_window_label)
    cv2.destroyWindow(canny_frame_window_label)
    cv2.destroyWindow(canny_hough_lines_window_label)

    # cv2.destroyAllWindows()


def draw_lines(img, lines, color=(0, 0, 255)):
    """
    Draw lines on an image
    """
    # The below for loop runs till r and theta values
    # are in the range of the 2d array
    for r_theta in lines:
        arr = numpy.array(r_theta[0], dtype=numpy.float64)
        r, theta = arr
        # Stores the value of cos(theta) in a
        a = numpy.cos(theta)

        # Stores the value of sin(theta) in b
        b = numpy.sin(theta)

        # x0 stores the value rcos(theta)
        x0 = a * r

        # y0 stores the value rsin(theta)
        y0 = b * r

        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 + 1000 * (-b))

        # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        y1 = int(y0 + 1000 * (a))

        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - 1000 * (-b))

        # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        y2 = int(y0 - 1000 * (a))

        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        # (0,0,255) denotes the colour of the line to be
        # drawn. In this case, it is red.
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


if __name__ == "__main__":

    def ijasd():
        """description"""
        from draugr.opencv_utilities import clean_up

        cleaned_images = []
        for a in (Path.home() / "Downloads" / "Vejbaner").iterdir():
            orig = cv2.imread(str(a))[:800, :800, :]
            cleaned_images.append(clean_up(orig))
        hough_line_calibrator(cleaned_images)

    ijasd()
