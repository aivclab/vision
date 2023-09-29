#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 16-06-2021
           """

from pathlib import Path

import cv2
import numpy
from draugr.opencv_utilities import get_video_sources, show_image
from warg import sink

vid_indices = iter(get_video_sources())

# print(vid_names)

swap = True

next(vid_indices)
next(vid_indices)

cam_right_d = cv2.VideoCapture(next(vid_indices))  # depth
next(vid_indices)
cam_left = cv2.VideoCapture(next(vid_indices))  # rgb
next(vid_indices)

cam_left_d = cv2.VideoCapture(next(vid_indices))  # depth
next(vid_indices)
can_right = cv2.VideoCapture(next(vid_indices))  # rgb
next(vid_indices)

# Reading the mapping values for stereo image rectification
cv_file = cv2.FileStorage(
    str(Path("../..") / "data" / "stereo_rectify_maps.xml"), cv2.FILE_STORAGE_READ
)
Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()

cv2.namedWindow("disp", cv2.WINDOW_NORMAL)
cv2.resizeWindow("disp", 600, 600)

cv2.createTrackbar("numDisparities", "disp", 1, 17, sink)
cv2.createTrackbar("blockSize", "disp", 5, 50, sink)
cv2.createTrackbar("preFilterType", "disp", 1, 1, sink)
cv2.createTrackbar("preFilterSize", "disp", 2, 25, sink)
cv2.createTrackbar("preFilterCap", "disp", 5, 62, sink)
cv2.createTrackbar("textureThreshold", "disp", 10, 100, sink)
cv2.createTrackbar("uniquenessRatio", "disp", 15, 100, sink)
cv2.createTrackbar("speckleRange", "disp", 0, 100, sink)
cv2.createTrackbar("speckleWindowSize", "disp", 3, 25, sink)
cv2.createTrackbar("disp12MaxDiff", "disp", 5, 25, sink)
cv2.createTrackbar("minDisparity", "disp", 5, 25, sink)

stereo = cv2.StereoBM_create()  # Creating an object of StereoBM algorithm

while True:
    # Capturing and storing left and right camera images
    retL, imgL = cam_left.read()
    retR, imgR = can_right.read()

    # retdL, imgdL = cam_left_d.read()
    retdR, imgdR = cam_right_d.read()

    if swap:
        retR, retL = retL, retR
        imgR, imgL = imgL, imgR

    if retL and retR:  # Proceed only if the frames have been captured
        imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

        left_rectified = cv2.remap(
            imgL_gray,
            Left_Stereo_Map_x,
            Left_Stereo_Map_y,
            cv2.INTER_LANCZOS4,
            cv2.BORDER_CONSTANT,
            0,
        )  # Applying stereo image rectification on the left image

        right_rectified = cv2.remap(
            imgR_gray,
            Right_Stereo_Map_x,
            Right_Stereo_Map_y,
            cv2.INTER_LANCZOS4,
            cv2.BORDER_CONSTANT,
            0,
        )  # Applying stereo image rectification on the right image

        numDisparities = cv2.getTrackbarPos("numDisparities", "disp") * 16
        blockSize = cv2.getTrackbarPos("blockSize", "disp") * 2 + 5
        preFilterType = cv2.getTrackbarPos("preFilterType", "disp")
        preFilterSize = cv2.getTrackbarPos("preFilterSize", "disp") * 2 + 5
        preFilterCap = cv2.getTrackbarPos("preFilterCap", "disp")
        textureThreshold = cv2.getTrackbarPos("textureThreshold", "disp")
        uniquenessRatio = cv2.getTrackbarPos("uniquenessRatio", "disp")
        speckleRange = cv2.getTrackbarPos("speckleRange", "disp")
        speckleWindowSize = cv2.getTrackbarPos("speckleWindowSize", "disp") * 2
        disp12MaxDiff = cv2.getTrackbarPos("disp12MaxDiff", "disp")
        minDisparity = cv2.getTrackbarPos("minDisparity", "disp")

        stereo.setNumDisparities(numDisparities)
        stereo.setBlockSize(blockSize)
        stereo.setPreFilterType(preFilterType)
        stereo.setPreFilterSize(preFilterSize)
        stereo.setPreFilterCap(preFilterCap)
        stereo.setTextureThreshold(textureThreshold)
        stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setSpeckleRange(speckleRange)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)

        disparity = stereo.compute(
            left_rectified, right_rectified
        )  # Calculating disparity using the StereoBM algorithm
        # NOTE: compute returns a 16bit signed single channel image,
        # CV_16S containing a disparity map scaled by 16. Hence it
        # is essential to convert it to CV_32F and scale it down 16 times.
        disparity = disparity.astype(numpy.float32)  # Converting to float32

        disparity = (
            disparity / 16.0 - minDisparity
        ) / numDisparities  # Scaling down the disparity values and normalizing them

        show_image(imgL)
        show_image(imgR)

        # show_image(imgdL)
        show_image(imgdR)

        if show_image(disparity, wait=1):
            break

print("Saving depth estimation parameters ......")

cv_file = cv2.FileStorage(
    "../data/depth_estimation_params_py.xml", cv2.FILE_STORAGE_WRITE
)
cv_file.write("numDisparities", numDisparities)
cv_file.write("blockSize", blockSize)
cv_file.write("preFilterType", preFilterType)
cv_file.write("preFilterSize", preFilterSize)
cv_file.write("preFilterCap", preFilterCap)
cv_file.write("textureThreshold", textureThreshold)
cv_file.write("uniquenessRatio", uniquenessRatio)
cv_file.write("speckleRange", speckleRange)
cv_file.write("speckleWindowSize", speckleWindowSize)
cv_file.write("disp12MaxDiff", disp12MaxDiff)
cv_file.write("minDisparity", minDisparity)
cv_file.write("M", 39.075)
cv_file.release()
