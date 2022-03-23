#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian"
__doc__ = r"""

           Created on 29/03/2020
           """

from pathlib import Path

import numpy
from IPython.display import IFrame
from matplotlib import pyplot

__all__ = ["plot_voxelgrid", "plot_points"]

with open(Path(__file__) / "templates" / "point_template.html", "r") as f:
    TEMPLATE_POINTS = f.readlines()

with open(Path(__file__) / "templates" / "voxelgrid_template.html", "r") as f:
    TEMPLATE_VG = f.readlines()


def plot_points(xyz, colors=None, size=0.1, axis=False):
    """

    Args:
      xyz:
      colors:
      size:
      axis:

    Returns:

    """
    positions = xyz.reshape(-1).tolist()

    camera_position = xyz.max(0) + abs(xyz.max(0))

    look = xyz.mean(0)

    if colors is None:
        colors = [1, 0.5, 0] * len(positions)

    elif len(colors.shape) > 1:
        colors = colors.reshape(-1).tolist()

    if axis:
        axis_size = xyz.ptp() * 1.5
    else:
        axis_size = 0

    with open("plot_points.html", "w") as html:
        html.write(
            TEMPLATE_POINTS.format(
                camera_x=camera_position[0],
                camera_y=camera_position[1],
                camera_z=camera_position[2],
                look_x=look[0],
                look_y=look[1],
                look_z=look[2],
                positions=positions,
                colors=colors,
                points_size=size,
                axis_size=axis_size,
            )
        )

    return IFrame("plot_points.html", width=800, height=800)


def plot_voxelgrid(v_grid, cmap="Oranges", show_axis: bool = False):
    """

    Args:
      v_grid:
      cmap:
      axis:

    Returns:

    """
    scaled_shape = v_grid.shape / min(v_grid.shape)

    # coordinates returned from argwhere are inversed so use [:, ::-1]
    points = numpy.argwhere(v_grid.vector)[:, ::-1] * scaled_shape

    s_m = pyplot.cm.ScalarMappable(cmap=cmap)
    rgb = s_m.to_rgba(v_grid.vector.reshape(-1)[v_grid.vector.reshape(-1) > 0])[:, :-1]

    camera_position = points.max(0) + abs(points.max(0))
    look = points.mean(0)

    if show_axis:
        axis_size = points.ptp() * 1.5
    else:
        axis_size = 0

    with open("plot_voxelgrid.html", "w") as html:
        html.write(
            TEMPLATE_VG.format(
                camera_x=camera_position[0],
                camera_y=camera_position[1],
                camera_z=camera_position[2],
                look_x=look[0],
                look_y=look[1],
                look_z=look[2],
                X=points[:, 0].tolist(),
                Y=points[:, 1].tolist(),
                Z=points[:, 2].tolist(),
                R=rgb[:, 0].tolist(),
                G=rgb[:, 1].tolist(),
                B=rgb[:, 2].tolist(),
                S_x=scaled_shape[0],
                S_y=scaled_shape[2],
                S_z=scaled_shape[1],
                n_voxels=sum(v_grid.vector.reshape(-1) > 0),
                axis_size=axis_size,
            )
        )

    return IFrame("plot_voxelgrid.html", width=800, height=800)
