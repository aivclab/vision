#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = ""

from pathlib import Path
from typing import Sequence, Union

import numpy
from matplotlib import pyplot
from matplotlib.colors import Colormap, LinearSegmentedColormap
from numpy import ndarray
from warg import Number


def discrete_cmap(
    N: int, base_cmap: Union[Colormap, str, None] = None
) -> LinearSegmentedColormap:
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return pyplot.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = pyplot.cm.get_cmap(base_cmap)
    color_list = base(numpy.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def scatter_plot_encoding_space(
    out_path: Path,
    mean: ndarray,
    log_var: ndarray,
    labels: Sequence,
    encoding_space_range: Number = 1,
    min_size_constant: Number = 2,
    N: int = 10,
):
    """

    :param out_path:
    :param mean:
    :param log_var:
    :param labels:
    :param encoding_space_range:
    :param min_size_constant:
    :param N:
    :return:"""
    sizes = numpy.abs(log_var.mean(-1)) + min_size_constant

    fig = pyplot.figure(figsize=(8, 6))
    pyplot.scatter(
        mean[:, 0],
        mean[:, 1],
        s=sizes,
        c=labels,
        marker="o",
        edgecolor="none",
        cmap=discrete_cmap(N, "jet"),
    )

    pyplot.colorbar(ticks=range(N))
    axes = pyplot.gca()

    axes.set_xlim([-encoding_space_range, encoding_space_range])
    axes.set_ylim([-encoding_space_range, encoding_space_range])

    pyplot.grid(True)
    pyplot.savefig(out_path)
    return fig
