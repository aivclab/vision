#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 03-05-2021
           """

import cv2
import dlib
from draugr.opencv_utilities import AsyncVideoStream
from draugr.opencv_utilities.dlib_utilities import (
    dlib68FacialLandmarksIndices,
    eye_aspect_ratio,
    shape_to_ndarray,
)


def aushdas():
    # define two constants, one for the eye aspect ratio to indicate
    # blink and then a second constant for the number of consecutive
    # frames the eye must be below the threshold
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 3
    upsample_num_times = 0
    # initialize the frame counters and the total number of blinks
    COUNTER = 0
    TOTAL = 0

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    for frame in AsyncVideoStream():
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for rect in detector(gray, upsample_num_times):  # loop over the face detections
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array

            eye_shape = shape_to_ndarray(predictor(gray, rect))
            left_eye = dlib68FacialLandmarksIndices.slice(
                eye_shape, dlib68FacialLandmarksIndices.left_eye
            )
            right_eye = dlib68FacialLandmarksIndices.slice(
                eye_shape, dlib68FacialLandmarksIndices.right_eye
            )

            cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)

            left_eye_ar = eye_aspect_ratio(left_eye)
            right_eye_ar = eye_aspect_ratio(right_eye)

            if (
                left_eye_ar < EYE_AR_THRESH or right_eye_ar < EYE_AR_THRESH
            ):  # check to see if the eye aspect ratio is below the blink    # threshold, and if so, increment
                # the blink frame counter
                COUNTER += 1
            else:  # otherwise, the eye aspect ratio is not below the blink    # threshold
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                COUNTER = 0

            cv2.putText(
                frame,
                f"Blinks: {TOTAL}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
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

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord(
            "q"
        ):  # if the `q` key was pressed, break from the loop
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()


if __name__ == "__main__":
    aushdas()
