#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 26/01/2022
           """

from enum import Enum

from sorcery import assigned_names


# TODO: WARNING NOTE ON ENUMS
# When you reload the module, you've effectively have two seperate types, and most types consider the type itself
# in  equality comparisons, e.g. if isinstance(other, Foo): return self.value == other.value else return NotImplemented.
# Importing relative and absolute is different!


class SelfAttentionTypeEnum(Enum):
    """ """

    (
        pairwise,  # pairwise subtraction 0
        patchwise,  # patchwise unfolding 1
    ) = assigned_names()


class PadModeEnum(Enum):
    """ """

    zero_pad, ref_pad = assigned_names()


if __name__ == "__main__":
    print({k: k.value for k in PadModeEnum.__iter__()})
