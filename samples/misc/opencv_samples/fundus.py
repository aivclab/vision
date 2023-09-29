#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 03/02/2022
           """

__all__ = []

from pathlib import Path

import cv2

# Load image, convert to grayscale, and find edges
from draugr.opencv_utilities import show_image

image = cv2.imread("1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

# Find contour and sort by contour area
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

ROI = None
# Find bounding box and extract ROI
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    ROI = image[y : y + h, x : x + w]
    break

show_image(ROI, wait=True, save_path=Path("ROI.png"))
