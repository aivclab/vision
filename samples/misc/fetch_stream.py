import cv2
from draugr.opencv_utilities import AsyncVideoStream

from samples.misc.exclude import VIDEO_SOURCE

#for i in AsyncVideoStream(VIDEO_SOURCE) :
#  print(i)
#  cv2.imshow("Output", i)

'''
import cv2
import numpy as np
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
vcap = cv2.VideoCapture("rtsp://192.168.1.2:5554/camera", cv2.CAP_FFMPEG)
while(1):
    ret, frame = vcap.read()
    if ret == False:
        print("Frame is empty")
        break
    else:
        cv2.imshow('VIDEO', frame)
        cv2.waitKey(1)
'''