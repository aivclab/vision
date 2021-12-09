#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 03-05-2021
           """

import cv2
import dlib
import numpy
from draugr.opencv_utilities import AsyncVideoStream, cv2_resize
from draugr.opencv_utilities.dlib.facealigner import align_face


def aushda():
    """

    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    face_size = (256, 256)
    upsample_num_times = 0

    print(type(detector))

    for frame_i, image in enumerate(AsyncVideoStream()):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = [cv2_resize(image.copy(), face_size)]

        for rect in detector(gray, upsample_num_times=upsample_num_times):
            faces.append(
                align_face(image, gray, rect, predictor, desired_face_size=face_size)
            )

        cv2.imshow("test", numpy.hstack(faces))

        if cv2.waitKey(1) & 0xFF == ord(
                "q"
        ):  # if the `q` key was pressed, break from the loop
            break


if __name__ == "__main__":
    aushda()
