#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 16-06-2021
           """

import time
from pathlib import Path

import cv2
from draugr.opencv_utilities import get_video_sources

vid_indices = iter(get_video_sources())

# print(vid_names)

swap = True

next(vid_indices)
next(vid_indices)

cam_right_d = cv2.VideoCapture(next(vid_indices))  # depth
next(vid_indices)
CamL = cv2.VideoCapture(next(vid_indices))  # rgb
next(vid_indices)

cam_left_d = cv2.VideoCapture(next(vid_indices))  # depth
next(vid_indices)
CamR = cv2.VideoCapture(next(vid_indices))  # rgb
next(vid_indices)

output_path = Path(".") / "data"

start = time.time()
T = 10
count = 0

while True:
    timer = T - int(time.time() - start)
    retR, frameR = CamR.read()
    retL, frameL = CamL.read()

    img1_temp = frameL.copy()

    cv2.putText(img1_temp, f"{timer!r}", (50, 50), 1, 5, (55, 0, 0), 5)
    cv2.imshow("imgR", frameR)
    cv2.imshow("imgL", img1_temp)

    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    retR, cornersR = cv2.findChessboardCorners(grayR, (9, 6), None)
    retL, cornersL = cv2.findChessboardCorners(grayL, (9, 6), None)

    if (
        (retR == True) and (retL == True) and timer <= 0
    ):  # If corners are detected in left and right image then we save it.
        count += 1
        cv2.imwrite(str(output_path / "stereoR" / f"img{count:d}.png"), frameR)
        cv2.imwrite(str(output_path / "stereoL" / f"img{count:d}.png"), frameL)

    if timer <= 0:
        start = time.time()

    # Press esc to exit
    if cv2.waitKey(1) & 0xFF == 27:
        print("Closing the cameras!")
        break

# Release the Cameras
CamR.release()
CamL.release()
cv2.destroyAllWindows()
