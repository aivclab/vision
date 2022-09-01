#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 01/08/2020
           """

__all__ = []


def test_import():
  import neodroidvision

  print(neodroidvision.__version__)


def test_package_data() -> None:
  import neodroidvision

  print(neodroidvision.PACKAGE_DATA_PATH / "Lato-Regular.ttf")


def test_import_regression():
  from neodroidvision import regression

  print(regression.__doc__)


def test_import_multitask():
  from neodroidvision import multitask

  print(multitask.__doc__)


def test_import_classification():
  from neodroidvision import classification

  print(classification.__doc__)


def test_import_segmentation():
  from neodroidvision import segmentation

  print(segmentation.__doc__)


def test_import_detection():
  from neodroidvision import detection

  print(detection.__doc__)


def test_import_utilities():
  from neodroidvision import utilities

  print(utilities.__doc__)


def test_import_data():
  from neodroidvision import data

  print(data.__doc__)
