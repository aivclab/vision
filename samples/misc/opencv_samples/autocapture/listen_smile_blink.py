#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 04-05-2021
           """

import cv2
import dlib  # pip install dlib -U
from _dlib_pybind11 import rectangle
from draugr.dlib_utilities import (
    Dlib68faciallandmarksindices,
    align_face,
    eye_aspect_ratio,
    mouth_aspect_ratio,
    shape_to_ndarray,
)
from draugr.opencv_utilities import AsyncVideoStream


def aushdas():
    """description"""
    cv2.namedWindow("test")
    cv2.namedWindow("rect")

    detector = dlib.get_frontal_face_detector()
    crude_predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
    detail_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    outer_upsample_num_times = 1
    inner_upsample_num_times = 0
    debug = True
    face_size = (256, 256)
    rect_aligned = rectangle(30, 30, 256 - 30, 256 - 30)

    for frame in AsyncVideoStream():
        gray_o = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for rect in detector(gray_o, outer_upsample_num_times):
            aligned = align_face(
                gray_o, gray_o, rect, crude_predictor, desired_face_size=face_size
            )
            # for rect_aligned in detector(aligned, inner_upsample_num_times): # alternative to hardcoded

            aligned_landmarks = shape_to_ndarray(
                detail_predictor(aligned, rect_aligned)
            )

            mouth = Dlib68faciallandmarksindices.slice(
                aligned_landmarks, Dlib68faciallandmarksindices.mouth
            )
            mouth_ar = mouth_aspect_ratio(mouth)

            left_eye = Dlib68faciallandmarksindices.slice(
                aligned_landmarks, Dlib68faciallandmarksindices.left_eye
            )
            left_eye_ar = eye_aspect_ratio(left_eye)

            right_eye = Dlib68faciallandmarksindices.slice(
                aligned_landmarks, Dlib68faciallandmarksindices.right_eye
            )
            right_eye_ar = eye_aspect_ratio(right_eye)

            visual_predictors = [mouth_ar, left_eye_ar, right_eye_ar]

            if debug:
                cv2.drawContours(aligned, [cv2.convexHull(mouth)], -1, (0, 255, 0), 1)
                cv2.drawContours(
                    aligned, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1
                )
                cv2.drawContours(
                    aligned, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1
                )

                cv2.imshow("rect", aligned)

            if False and debug:
                cv2.putText(
                    frame,
                    f"mar: {mouth_ar}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Left eye_AR: {left_eye_ar:.2f}",
                    (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Right eye_AR: {right_eye_ar:.2f}",
                    (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        cv2.imshow("test", frame)
        if cv2.waitKey(1) & 0xFF == ord(
            "q"
        ):  # if the `q` key was pressed, break from the loop
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()


if __name__ == "__main__":
    aushdas()
