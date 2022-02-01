#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 27/01/2022
           """

from warg import Number


def eye_focal_point(
    crescent_width_cm,
    pupil_radius_cm: Number,
    source_offset_cm: Number,
    camera_to_subject_distance_cm: Number,
) -> Number:
    """
    crescent width to eye depth

     For a crescent to be visible, the refractive error must exceed a critical magnitude (the dead zone). If 2,., e,
     and d are held constant, the crescent width s will expand rapidly as the refractive error in diopters increases
     until a critical value is reached at ±5 D, beyond which changes in crescent width become Ie s perceptible.

    """

    # A = 100 / x_solve_for
    # myopic = (camera_to_subject_distance*(-A-(1/camera_to_subject_distance)))
    x_myopic = (
        100 * camera_to_subject_distance_cm * (crescent_width_cm - 2 * pupil_radius_cm)
    ) / (source_offset_cm + 2 * pupil_radius_cm - crescent_width_cm)
    # and o + 2 p!=s and 2
    # c o p!=c o s

    # hyperopic = (camera_to_subject_distance*(A+(1/camera_to_subject_distance)))
    x_hyperopic = (
        100 * camera_to_subject_distance_cm * (2 * pupil_radius_cm - crescent_width_cm)
    ) / (source_offset_cm - 2 * pupil_radius_cm + crescent_width_cm)
    # and o + s != 2 p and 2  c  o  p != c  o  s

    # s = 2*pupil_radius-source_offset/myopic

    return x_myopic, x_hyperopic


def refractive_error():
    pass


if __name__ == "__main__":
    print(eye_focal_point(0.02, 0.4, 1, 100))
