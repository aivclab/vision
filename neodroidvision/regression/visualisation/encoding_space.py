#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Christian Heider Nielsen'
__doc__ = ''

from matplotlib import pyplot
import numpy


def scatter_plot_encoding_space(out_path,
                                mean,
                                log_var,
                                labels,
                                encoding_space_range=1,
                                min_size_constant=2,
                                N=10):
  sizes = numpy.abs(log_var.mean(-1)) + min_size_constant

  pyplot.figure(figsize=(8, 6))
  pyplot.scatter(mean[:, 0],
                 mean[:, 1],
                 s=sizes,
                 c=labels,
                 marker='o',
                 edgecolor='none',
                 cmap=discrete_cmap(N, 'jet'))

  pyplot.colorbar(ticks=range(N))
  axes = pyplot.gca()

  axes.set_xlim([-encoding_space_range, encoding_space_range])
  axes.set_ylim([-encoding_space_range, encoding_space_range])

  pyplot.grid(True)
  pyplot.savefig(out_path)
  pyplot.clf()


def discrete_cmap(N, base_cmap=None):
  """Create an N-bin discrete colormap from the specified input map"""

  # Note that if base_cmap is a string or None, you can simply do
  #    return pyplot.cm.get_cmap(base_cmap, N)
  # The following works for string, None, or a colormap instance:

  base = pyplot.cm.get_cmap(base_cmap)
  color_list = base(numpy.linspace(0, 1, N))
  cmap_name = base.name + str(N)
  return base.from_list(cmap_name, color_list, N)
