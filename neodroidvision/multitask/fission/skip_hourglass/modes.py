#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 02/03/2020
           """

from enum import Enum

__all__ = ["UpscaleMode", "MergeMode"]


class MergeMode(Enum):
    Concat = 0
    Add = 1


class UpscaleMode(Enum):
    FractionalTranspose = 0
    Upsample = 1


if __name__ == "__main__":
    assert MergeMode.Concat in MergeMode

    assert not (UpscaleMode.Upsample in MergeMode)

    assert UpscaleMode.Upsample in UpscaleMode

    assert not (MergeMode.Add in UpscaleMode)

    # assert not (0 in UpscaleMode)

    # assert not (3 in UpscaleMode)
