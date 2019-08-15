#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'cnheider'
__doc__ = ''

import matplotlib.pyplot as plt
import numpy as np


def scatter_plot_encoding_space(out_path,
                                mean,
                                log_var,
                                labels,
                                encoding_space_range=1,
                                N=10):
  sizes = np.abs(log_var.mean(-1))

  plt.figure(figsize=(8, 6))
  plt.scatter(mean[:, 0],
              mean[:, 1],
              s=sizes,
              c=labels,
              marker='o',
              edgecolor='none',
              cmap=discrete_cmap(N, 'jet'))

  plt.colorbar(ticks=range(N))
  axes = plt.gca()

  axes.set_xlim([-encoding_space_range, encoding_space_range])
  axes.set_ylim([-encoding_space_range, encoding_space_range])

  plt.grid(True)
  plt.savefig(out_path)
  plt.clf()


def discrete_cmap(N, base_cmap=None):
  """Create an N-bin discrete colormap from the specified input map"""

  # Note that if base_cmap is a string or None, you can simply do
  #    return plt.cm.get_cmap(base_cmap, N)
  # The following works for string, None, or a colormap instance:

  base = plt.cm.get_cmap(base_cmap)
  color_list = base(np.linspace(0, 1, N))
  cmap_name = base.name + str(N)
  return base.from_list(cmap_name, color_list, N)
