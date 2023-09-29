#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 03-05-2021
           """

from itertools import cycle

import cv2
import dlib
from draugr.dlib_utilities import Dlib68faciallandmarksindices
from draugr.opencv_utilities import AsyncVideoStream

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def visualize_facial_landmarks(
    image,
    shape,
    colors=(
        (19, 199, 109),
        (79, 76, 240),
        (230, 159, 23),
        (168, 100, 168),
        (158, 163, 32),
        (163, 38, 32),
        (180, 42, 220),
    ),
    alpha=0.75,
):
    """

    Args:
      image:
      shape:
      colors:
      alpha:

    Returns:

    """
    # create two copies of the input image -- one for the
    # overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()

    colors = iter(cycle(colors))

    for (
        name
    ) in (
        Dlib68faciallandmarksindices
    ):  # loop over the facial landmark regions individually
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = name.value
        pts = shape[j:k]

        col = next(colors)

        # check if are supposed to draw the jawline
        if name == Dlib68faciallandmarksindices.jaw:
            # since the jawline is a non-enclosed facial region,
            # just draw lines between the (x, y)-coordinates
            for l in range(1, len(pts)):
                pt_a = tuple(pts[l - 1])
                pt_b = tuple(pts[l])
                cv2.line(overlay, pt_a, pt_b, col, 2)

        # otherwise, compute the convex hull of the facial
        # landmark coordinates points and display it
        else:
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], -1, col, -1)

    # apply the transparent overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # return the output image
    return output


if __name__ == "__main__":

    def asijdas():
        """description"""
        upsample = 0
        for image in AsyncVideoStream():
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            for i, rect in enumerate(detector(gray, upsample)):
                # determine the facial landmarks for the face region, then
                # convert the landmark (x, y)-coordinates to a NumPy array
                shape = shape_to_ndarray(predictor(gray, rect))

                if False:
                    for (
                        name
                    ) in (
                        Dlib68faciallandmarksindices
                    ):  # loop over the face parts individually
                        # clone the original image so we can draw on it, then
                        # display the name of the face part on the image
                        (i, j) = name.value
                        clone = image.copy()
                        cv2.putText(
                            clone,
                            str(name),
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )

                        # loop over the subset of facial landmarks, drawing the
                        # specific face part
                        for x, y in shape[i:j]:
                            cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

                        if False:
                            # extract the ROI of the face region as a separate image
                            (x, y, w, h) = cv2.boundingRect(numpy.array([shape[i:j]]))

                            # show the particular face part
                            cv2.imshow(
                                "ROI",
                                cv2_resize(
                                    image[y : y + h, x : x + w],
                                    size=(250, 250),
                                    inter=cv2.INTER_CUBIC,
                                ),
                            )
                            cv2.imshow("Image", clone)

                cv2.imshow("Image", visualize_facial_landmarks(image, shape))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    asijdas()
