#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 03-05-2021
           """

import cv2
import dlib

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
from draugr.opencv_utilities import AsyncVideoStream, show_image
from draugr.opencv_utilities.dlib_utilities import shape_to_ndarray

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)
upsample_num_times = 0

for image in AsyncVideoStream():
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for (i, rect) in enumerate(detector(gray, upsample_num_times)):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape_to_ndarray(predictor(gray, rect)):
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    if show_image(
        image, wait=5, char="q"
    ):  # show the output image with the face detections + facial landmarks
        break

cv2.destroyAllWindows()
cap.release()
