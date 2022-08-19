#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian"
__doc__ = r"""

           Created on 29/03/2020
           """

import numpy
from matplotlib import pyplot

from .plot3d import plot_voxelgrid

__all__ = ["VoxelGrid"]


class VoxelGrid(object):
    """description"""

    def __init__(self, points, x_y_z=(1, 1, 1), bb_cuboid=True, build=True):
        """
        Parameters
        ----------
        points: (N,3) ndarray
                The point cloud from wich we want to construct the VoxelGrid.
                Where N is the number of points in the point cloud and the second
                dimension represents the x, y and z coordinates of each point.

        x_y_z:  list
                The segments in wich each axis will be divided.
                x_y_z[0]: x axis
                x_y_z[1]: y axis
                x_y_z[2]: z axis

        bb_cuboid(Optional): bool
                If True(Default):
                    The bounding box of the point cloud will be adjusted
                    in order to have all the dimensions of equal lenght.
                If False:
                    The bounding box is allowed to have dimensions of different sizes.
        """
        self.points = points

        xyz_min = numpy.min(points, axis=0) - 0.001
        xyz_max = numpy.max(points, axis=0) + 0.001

        if bb_cuboid:
            #: adjust to obtain a  minimum bounding box with all sides of equal lenght
            diff = max(xyz_max - xyz_min) - (xyz_max - xyz_min)
            xyz_min = xyz_min - diff / 2
            xyz_max = xyz_max + diff / 2

        self.xyz_min = xyz_min
        self.xyz_max = xyz_max

        segments = []
        shape = []

        for i in range(3):
            # note the +1 in num
            if type(x_y_z[i]) is not int:
                raise TypeError(f"x_y_z[{i}] must be int")
            s, step = numpy.linspace(
                xyz_min[i], xyz_max[i], num=(x_y_z[i] + 1), retstep=True
            )
            segments.append(s)
            shape.append(step)

        self.segments = segments

        self.shape = shape

        self.n_voxels = x_y_z[0] * x_y_z[1] * x_y_z[2]
        self.n_x = x_y_z[0]
        self.n_y = x_y_z[1]
        self.n_z = x_y_z[2]

        self.id = f"{x_y_z[0]},{x_y_z[1]},{x_y_z[2]}-{bb_cuboid}"

        if build:
            self.build()

    def build(self):
        """description"""
        structure = numpy.zeros((len(self.points), 4), dtype=int)

        structure[:, 0] = numpy.searchsorted(self.segments[0], self.points[:, 0]) - 1

        structure[:, 1] = numpy.searchsorted(self.segments[1], self.points[:, 1]) - 1

        structure[:, 2] = numpy.searchsorted(self.segments[2], self.points[:, 2]) - 1

        # i = ((y * n_x) + x) + (z * (n_x * n_y))
        structure[:, 3] = ((structure[:, 1] * self.n_x) + structure[:, 0]) + (
            structure[:, 2] * (self.n_x * self.n_y)
        )

        self.structure = structure

        vector = numpy.zeros(self.n_voxels)
        count = numpy.bincount(self.structure[:, 3])
        vector[: len(count)] = count

        self.vector = vector.reshape(self.n_z, self.n_y, self.n_x)

    def plot(self, d=2, cmap="Oranges", show_axis: bool = False):
        """

        Args:
          d:
          cmap:
          axis:

        Returns:
        :param show_axis:
        :type show_axis:

        """
        if d == 2:

            fig, axes = pyplot.subplots(
                int(numpy.ceil(self.n_z / 4)), 4, figsize=(8, 8)
            )

            pyplot.tight_layout()

            for i, ax in enumerate(axes.flat):
                if i >= len(self.vector):
                    break
                im = ax.imshow(self.vector[i], cmap=cmap, interpolation="none")
                ax.set_title("Level " + str(i))

            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label("NUMBER OF POINTS IN VOXEL")

        elif d == 3:
            return plot_voxelgrid(self, cmap=cmap, show_axis=show_axis)
