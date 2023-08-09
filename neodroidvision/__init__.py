#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os
from importlib import resources
from importlib.metadata import PackageNotFoundError
from warnings import warn


from apppath import AppPath
from warg import package_is_editable

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


from typing import Any


PROJECT_NAME = __project__.lower().strip().replace(" ", "_")
PROJECT_VERSION = __version__
PROJECT_YEAR = 2018
PROJECT_AUTHOR = __author__.lower().strip().replace(" ", "_")
PROJECT_ORGANISATION = "Neodroid"
PROJECT_APP_PATH = AppPath(app_name=PROJECT_NAME, app_author=PROJECT_AUTHOR)
INCLUDE_PROJECT_READMES = False

PACKAGE_DATA_PATH = resources.files(PROJECT_NAME) / "data"

try:
    DEVELOP = package_is_editable(PROJECT_NAME)
except PackageNotFoundError as e:
    DEVELOP = True


def get_version(append_time: Any = DEVELOP) -> str:
    """

    Args:
      append_time:

    Returns:

    """
    version = __version__
    if not version:
        version = os.getenv("VERSION", "0.0.0")

    if append_time:
        now = datetime.datetime.utcnow()
        date_version = now.strftime("%Y%m%d%H%M%S")
        # date_version = time.time()

        if version:
            # Most git tags are prefixed with 'v' (example: v1.2.3) this is
            # never desirable for artifact repositories, so we strip the
            # leading 'v' if it's present.
            version = (
                version[1:]
                if isinstance(version, str) and version.startswith("v")
                else version
            )
        else:
            # Default version is an ISO8601 compliant datetime. PyPI doesn't allow
            # the colon ':' character in its versions, and time is required to allow
            # for multiple publications to master in one day. This datetime string
            # uses the 'basic' ISO8601 format for both its date and time components
            # to avoid issues with the colon character (ISO requires that date and
            # time components of a date-time string must be uniformly basic or
            # extended, which is why the date component does not have dashes.
            #
            # Publications using datetime versions should only be made from master
            # to represent the HEAD moving forward.
            warn(
                f"Environment variable VERSION is not set, only using datetime: {date_version}"
            )

            # warn(f'Environment variable VERSION is not set, only using timestamp: {version}')

        version = f"{version}.{date_version}"

    return version


if __version__ is None:
    __version__ = get_version(append_time=True)

__version_info__ = tuple(int(segment) for segment in __version__.split("."))
