#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from matplotlib import pyplot
import numpy
import cv2
import glob

from pynput import keyboard
from drawing_utilities import draw_cube, draw_axis

from neodroidvision import PROJECT_APP_PATH

__author__ = 'Christian Heider Nielsen'
__doc__ = r'''

           Created on 09/10/2019
           '''

calibration_file_name = str(PROJECT_APP_PATH.user_data / 'camera_calibration.npz')
intersections_shape = (9, 6)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
checkerboard_points = numpy.zeros((numpy.prod(intersections_shape), 3), numpy.float32)
checkerboard_points[:, :2] = numpy.mgrid[0:intersections_shape[0],
                             0:intersections_shape[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
object_points = []  # 3d point in real world space
img_points = []  # 2d points in image plane.

base = Path.home() / 'Data' / 'Opencv' / 'images' / 'left'

save_keys = ('mtx', 'dist', 'rvecs', 'tvecs')


def image_loader_generator():
  images_2 = glob.glob(f'{base}/*.jpg')
  for fname in images_2:
    yield cv2.imread(fname)


a = ''


def on_press(key):
  global a
  try:
    a = key
  except AttributeError:
    pass


def on_release(key):
  if key == keyboard.Key.esc:
    return False


with keyboard.Listener(
  on_press=on_press,
  on_release=on_release) as listener:  # Collect events until released

  def webcam_generator():
    global a
    cap = cv2.VideoCapture(0)

    while True:
      ret, frame = cap.read()

      if a != '':
        if a == 'q':
          break

        yield frame

      a = ''

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    raise StopIteration


  # images = image_loader_generator()
  images = webcam_generator()


  def find_intersections():
    for img, _ in zip(images, range(10)):
      gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

      # Find the chess board corners
      intrsc_found, intersections = cv2.findChessboardCorners(gray_img, intersections_shape, None)

      if intrsc_found:  # If found, add object points, image points (after refining them)
        object_points.append(checkerboard_points)

        sub_intrsc = cv2.cornerSubPix(gray_img, intersections, (11, 11), (-1, -1), criteria)
        img_points.append(sub_intrsc)

        # Draw and display the corners
        pyplot.imshow(cv2.drawChessboardCorners(img, intersections_shape, sub_intrsc, intrsc_found))
        pyplot.show()


  def calibrate():
    img = cv2.imread(str(base / 'left12.jpg'))
    h, w = img.shape[:2]
    shape_ = (w, h)
    (intrsc_found,
     camera_mtx,
     dist_coef,
     rot_vecs,
     trans_vecs) = cv2.calibrateCamera(object_points,
                                       img_points,
                                       shape_,
                                       None,
                                       None)

    numpy.savez(calibration_file_name,
                **{b:i for b, i in zip(save_keys, [camera_mtx,
                                                   dist_coef,
                                                   rot_vecs,
                                                   trans_vecs])})

    '''
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_mtx,
                                                        dist_coef,
                                                        shape_,
                                                        1,
                                                        shape_)
  
    # This is the shortest path. Just call the function and use ROI obtained above to crop the result.
    dst = cv2.undistort(img, camera_mtx, dist_coef, None, new_camera_mtx)
  
    #This is curved path. First find a mapping function from distorted image to undistorted image. Then use 
    the remap function.
    #mapx,mapy = cv2.initUndistortRectifyMap(camera_mtx,dist_coef,None,new_camera_mtx,(w,h),5)
    #dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
  
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]   # crop the image
  
    pyplot.imshow(dst)
    pyplot.show()
  
    tot_error = 0
    for i in range(len(object_points)):
      point_projections, _ = cv2.projectPoints(object_points[i],
                                               rot_vecs[i],
                                               trans_vecs[i],
                                               camera_mtx,
                                               dist_coef)
      error = cv2.norm(img_points[i],
                       point_projections,
                       cv2.NORM_L2) / len(point_projections)
      tot_error += error
  
    print(f"total error:{tot_error / len(object_points)}")
  
    '''


  def load_and_draw():
    # Load previously saved data
    with numpy.load(calibration_file_name) as X:
      camera_mtx, dist_coef, _, _ = [X[i] for i in save_keys]

    for img in images:

      gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      intrsc_found, intersections = cv2.findChessboardCorners(gray_img, intersections_shape, None)

      if intrsc_found:
        sub_intrsc = cv2.cornerSubPix(gray_img,
                                      intersections,
                                      winSize=(11, 11),
                                      zeroZone=(-1, -1),
                                      criteria=criteria)

        # Find the rotation and translation vectors.
        _, rot_vecs, trans_vecs, inliers = cv2.solvePnPRansac(checkerboard_points,
                                                              sub_intrsc,
                                                              camera_mtx,
                                                              dist_coef)

        img = draw_cube(img,
                        rot_vecs,
                        trans_vecs,
                        camera_mtx,
                        dist_coef)
        pyplot.imshow(img)
        pyplot.show()


  find_intersections()
  calibrate()
  load_and_draw()
