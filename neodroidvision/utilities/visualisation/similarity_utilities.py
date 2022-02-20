#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 27/07/2020
           """

__all__ = ["boxed_text_overlay_plot"]

import numpy
from matplotlib import pyplot
from torch import Tensor


def boxed_text_overlay_plot(img: Tensor, text: str) -> None:
    """

    :param img:
    :type img:
    :param text:
    :type text:"""
    npimg = img.numpy()
    pyplot.axis("off")
    if text is not None:
        pyplot.text(
            x=0,
            y=0,
            s=text,
            style="italic",
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 10},
        )
    pyplot.imshow(numpy.transpose(npimg, (1, 2, 0)))
    pyplot.show()
