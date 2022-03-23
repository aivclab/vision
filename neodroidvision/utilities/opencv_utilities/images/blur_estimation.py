#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 03/02/2022
           """

__all__ = ["is_picture_blurry"]

import cv2


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


def is_picture_blurry(image, threshold=100) -> bool:
    return variance_of_laplacian(image) < threshold


if __name__ == "__main__":
    for ip in ("right.jpg", "left.jpg", "ok.jpg"):
        print(is_picture_blurry(cv2.imread(f"../exclude/{ip}"), 50))
