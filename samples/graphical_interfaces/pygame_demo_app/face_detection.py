#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'cnheider'
__doc__ = r'''
           '''

import time

import cv2
import pygame
from pygame import camera

FACE_HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_mcs_righteye.xml")
NOSE_HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_mcs_nose.xml")
MOUTH_HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_mcs_mouth.xml")

# Screen settings
SCREEN = [640, 360]


def surface_to_string(surface):
  """Convert a pygame surface into string"""
  return pygame.image.tostring(surface, 'RGB')


def pygame_to_cvimage(surface):
  """Convert a pygame surface into a cv image"""
  cv_image = cv2.CreateImageHeader(surface.get_size(), cv2.IPL_DEPTH_8U, 3)
  image_string = surface_to_string(surface)
  cv2.SetData(cv_image, image_string)
  return cv_image


def cvimage_grayscale(cv_image):
  """Converts a cvimage into grayscale"""
  grayscale = cv2.CreateImage(cv2.GetSize(cv_image), 8, 1)
  cv2.CvtColor(cv_image, grayscale, cv2.COLOR_RGB2GRAY)
  return grayscale


def cvimage_to_pygame(image):
  """Convert cvimage into a pygame image"""
  image_rgb = cv2.CreateMat(image.height, image.width, cv2.CV_8UC3)
  cv2.CvtColor(image, image_rgb, cv2.COLOR_BGR2RGB)
  return pygame.image.frombuffer(image.tostring(),
                                 cv2.GetSize(image_rgb),
                                 "RGB")


def detect_faces(cv_image):
  """Detects faces based on haar. Returns points"""
  return FACE_HAAR.detectMultiScale(cvimage_grayscale(cv_image))


def detect_eyes(cv_image):
  """Detects eyes based on haar. Returns points"""
  return EYE_HAAR.detectMultiScale(cvimage_grayscale(cv_image))


def detect_nose(cv_image):
  """Detects nose based on haar. Returns ponts"""
  return NOSE_HAAR.detectMultiScale(cvimage_grayscale(cv_image))


def detect_mouth(cv_image):
  """Detects mouth based on haar. Returns points"""
  return MOUTH_HAAR.detectMultiScale(cvimage_grayscale(cv_image))


def draw_from_points(cv_image, points):
  """Takes the cv_image and points and draws a rectangle based on the points.
  Returns a cv_image."""
  for (x, y, w, h), n in points:
    cv2.rectangle(cv_image, (x, y), (x + w, y + h), 255)
  return cv_image


if __name__ == '__main__':

  # Set game screen
  screen = pygame.display.set_mode(SCREEN)

  pygame.init()  # Initialize pygame
  camera.init()  # Initialize camera

  # Load camera source then start
  cam = camera.Camera('/dev/video0', SCREEN)
  cam.start()

  while 1:  # Ze loop

    time.sleep(1 / 120)  # 60 frames per second

    image = cam.get_image()  # Get current webcam image

    cv_image = pygame_to_cvimage(image)  # Create cv image from pygame image

    points = (detect_eyes(cv_image) +
              detect_nose(cv_image) +
              detect_mouth(cv_image) +
              detect_faces(cv_image))  # Get points of faces.

    cv_image = draw_from_points(cv_image, points)  # Draw points

    screen.fill([0, 0, 0])  # Blank fill the screen

    screen.blit(cvimage_to_pygame(cv_image), (0, 0))  # Load new image on screen

    pygame.display.update()  # Update pygame display
