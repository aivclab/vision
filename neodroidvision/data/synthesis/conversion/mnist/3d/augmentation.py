#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian"
__doc__ = r"""

           Created on 29/03/2020
           """

import numpy

__all__ = ["rotate_y", "rotate_x", "rotate_z"]

from warg import Number


def rotate_y(angle: Number, degrees: bool = True) -> numpy.ndarray:
    """Ry ,rotate along y-axis"""
    if degrees:
        angle = numpy.deg2rad(angle)

    cy = numpy.cos(angle)
    sy = numpy.sin(angle)

    return numpy.array([[cy, 0, -sy, 0], [0, 1, 0, 0], [sy, 0, cy, 0], [0, 0, 0, 1]])


def rotate_x(angle: Number, degrees: bool = True) -> numpy.ndarray:
    """Rx ,rotate along x-axis"""
    if degrees:
        angle = numpy.deg2rad(angle)

    cx = numpy.cos(angle)
    sx = numpy.sin(angle)

    return numpy.array([[1, 0, 0, 0], [0, cx, sx, 0], [0, -sx, cx, 0], [0, 0, 0, 1]])


def rotate_z(angle: Number, degrees: bool = True) -> numpy.ndarray:
    """Rz ,rotate along z-axis"""
    if degrees:
        angle = numpy.deg2rad(angle)

    cz = numpy.cos(angle)
    sz = numpy.sin(angle)

    return numpy.array([[cz, sz, 0, 0], [-sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
