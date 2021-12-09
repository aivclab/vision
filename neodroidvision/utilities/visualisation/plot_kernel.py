#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 29/07/2020
           """

__all__ = ["plot_kernels"]

from matplotlib import pyplot


def plot_kernels(tensor, number_cols=5, m_interpolation="bilinear"):
  """
  Function to visualize the kernels.

  Arguments:
    tensor:
    number_cols: number of columns to be displayed
    m_interpolation: interpolation methods matplotlib. See in:

    https://matplotlib.org/gallery/images_contours_and_fields/interpolation_methods.html"""

  number_kernels = tensor.shape[0]
  number_rows = 1 + number_kernels // number_cols
  fig = pyplot.figure(figsize=(number_cols, number_rows))
  for i in range(number_kernels):
    ax1 = fig.add_subplot(number_rows, number_cols, i + 1)
    ax1.imshow(tensor[i][0, :, :], interpolation=m_interpolation, cmap="gray")
    ax1.axis("off")
