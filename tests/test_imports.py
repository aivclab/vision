#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 01/08/2020
           """

__all__ = []

from pathlib import Path

import pkg_resources
from neodroidvision import PROJECT_NAME


def test_import():
    import neodroidvision

    print(neodroidvision.__version__)

def test_package_data()->None:
    import neodroidvision
    print(neodroidvision.PACKAGE_DATA_PATH/"Lato-Regular.ttf")