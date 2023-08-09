#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
            pip install opencv-python opencv_contrib_python opencv-contrib-python -U

           """

import time

import cv2
import numpy
import pygame
from pygame import camera

FACE_HAAR = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
EYE_HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
NOSE_HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_nose.xml")
MOUTH_HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_mouth.xml")

# Screen settings
SCREEN = [640, 360]

try:
    from cv2 import CreateImageHeader
except:
    pass
    # raise ModuleNotFoundError("Try: pip install opencv-python opencv-contrib-python -U")


def surface_to_numpy(surface) -> numpy.ndarray:
    """Convert a pygame surface into string"""
    return pygame.surfarray.pixels3d(surface)


def rgb_to_grayscale(cv_image, rgb_weights=(0.2989, 0.5870, 0.1140)):
    """Converts a cvimage into grayscale"""
    return numpy.dot(cv_image[..., :3], rgb_weights).astype(numpy.uint8)


def cvimage_to_pygame(image):
    """Convert cvimage into a pygame image"""
    return pygame.surfarray.make_surface(image)
    # return pygame.image.frombuffer(image.tostring(), image.size, "RGB")


def detect_faces(cv_image):
    """Detects faces based on haar. Returns points"""
    return FACE_HAAR.detectMultiScale(rgb_to_grayscale(cv_image))


def detect_eyes(cv_image):
    """Detects eyes based on haar. Returns points"""
    return EYE_HAAR.detectMultiScale(rgb_to_grayscale(cv_image))


def detect_nose(cv_image):
    """Detects nose based on haar. Returns ponts"""
    return NOSE_HAAR.detectMultiScale(rgb_to_grayscale(cv_image))


def detect_mouth(cv_image):
    """Detects mouth based on haar. Returns points"""
    return MOUTH_HAAR.detectMultiScale(rgb_to_grayscale(cv_image))


def draw_from_points(cv_image, points):
    """Takes the cv_image and points and draws a rectangle based on the points.
    Returns a cv_image."""
    cv_image = numpy.ascontiguousarray(cv_image, dtype=numpy.uint8)
    for f in points:
        for x, y, w, h in f:
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), 255)
    return cv_image


if __name__ == "__main__":
    # Set game screen
    screen = pygame.display.set_mode(SCREEN)

    pygame.init()  # Initialize pygame
    camera.init()  # Initialize camera
    cameras = pygame.camera.list_cameras()
    # Load camera source then start
    cam = camera.Camera(cameras[0], SCREEN)
    cam.start()

    while 1:  # Ze loop
        time.sleep(1 / 120)  # 60 frames per second

        image = cam.get_image()  # Get current webcam image

        cv_image = surface_to_numpy(image)  # Create cv image from pygame image

        points = (
            detect_eyes(cv_image),
            # detect_nose(cv_image),
            # detect_mouth(cv_image),
            detect_faces(cv_image),
        )  # Get points of faces.

        cv_image = draw_from_points(cv_image, points)  # Draw points

        screen.fill([0, 0, 0])  # Blank fill the screen

        screen.blit(cvimage_to_pygame(cv_image), (0, 0))  # Load new image on screen

        pygame.display.update()  # Update pygame display
