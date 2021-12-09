#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/03/2020
           """

from abc import ABC

from draugr.torch_utilities import SupervisedDataset

__all__ = ["ObjectDetectionDataset"]

from warg import drop_unused_kws


class ObjectDetectionDataset(SupervisedDataset, ABC):
    """ """

    categories = None

    @drop_unused_kws
    def __init__(self):
        super().__init__()
