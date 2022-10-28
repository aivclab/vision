#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 5/5/22
           """

from torchvision import transforms

__all__ = ["Replicate"]

Replicate = lambda n: transforms.Lambda(lambda x: x.repeat(n, 1, 1))
