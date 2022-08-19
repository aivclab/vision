#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 14-02-2021
           """

__all__ = ["unstandardise_image", "preprocess_image", "overlay_cam_on_image"]

import cv2
import numpy
from torchvision import transforms


def overlay_cam_on_image(img, mask):
    """

    Args:
      img:
      mask:

    Returns:

    """
    heatmap = cv2.applyColorMap(numpy.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = numpy.float32(heatmap) / 255
    cam = heatmap + numpy.float32(img)
    cam /= numpy.max(cam) + 1e-5
    return numpy.uint8(255 * cam)


def preprocess_image(img):
    """Preprocesses it for VGG19"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    preprocessing = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    return preprocessing(img.copy()).unsqueeze(0)


def unstandardise_image(img):
    """

    Args:
      img:

    Returns:

    """
    img = img - numpy.mean(img)
    img /= numpy.std(img) + 1e-5
    return numpy.uint8(numpy.clip(img * 0.1 + 0.5, 0, 1) * 255)
