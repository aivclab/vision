#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 01/08/2020
           """

__all__ = []

import pytest


def test_import():
    import neodroidvision

    print(neodroidvision.__version__)


def test_package_data() -> None:
    import neodroidvision

    print(neodroidvision.PACKAGE_DATA_PATH / "Lato-Regular.ttf")


def test_import_regression():
    from neodroidvision import regression
    from neodroidvision import multitask
    from neodroidvision import classification
    from neodroidvision import segmentation
    from neodroidvision import detection
    from neodroidvision import data

    print(regression.__doc__)
    print(multitask.__doc__)
    print(classification.__doc__)
    print(segmentation.__doc__)
    print(detection.__doc__)
    print(data.__doc__)
