#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from apppath import AppPath

try:
    from importlib.resources import files
    from importlib.metadata import PackageNotFoundError
except (ModuleNotFoundError, ImportError) as e:
    from importlib_metadata import PackageNotFoundError
    from importlib_resources import files

from warg import package_is_editable, clean_string, get_version


__project__ = "NeodroidVision"
__author__ = "Christian Heider Nielsen"
__version__ = "0.3.0"
__doc__ = r"""
.. module:: neodroidvision
   :platform: Unix, Windows
   :synopsis: A set of general computer vision tools build for the neodroid platform.

.. moduleauthor:: Christian Heider Nielsen <christian.heider@alexandra.dk>

Created on 27/04/2019

@author: cnheider
"""

# __all__ = ['PROJECT_APP_PATH', 'PROJECT_NAME', 'PROJECT_VERSION', 'get_version']


PROJECT_NAME = clean_string(__project__)
PROJECT_VERSION = __version__
PROJECT_YEAR = 2018
PROJECT_AUTHOR = clean_string(__author__)
PROJECT_ORGANISATION = clean_string("Neodroid")
PROJECT_APP_PATH = AppPath(app_name=PROJECT_NAME, app_author=PROJECT_AUTHOR)
INCLUDE_PROJECT_READMES = False

PACKAGE_DATA_PATH = files(PROJECT_NAME) / "data"

try:
    DEVELOP = package_is_editable(PROJECT_NAME)
except PackageNotFoundError as e:
    DEVELOP = True


__version__ = get_version(__version__, append_time=DEVELOP)
__version_info__ = tuple(int(segment) for segment in __version__.split("."))
