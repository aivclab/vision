#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 29/03/2020
           """

from typing import Tuple

__all__ = ["CamVid"]

from draugr.torch_utilities import SupervisedDataset


class CamVid(SupervisedDataset):
    """ """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @property
    def predictor_shape(self) -> Tuple[int, ...]:
        """ """
        return self.image_size

    @property
    def response_shape(self) -> Tuple[int, ...]:
        """ """
        return (self.response_channels,)

    predictor_channels = 3  # RGB input
    colors_dict = {
        (64, 128, 64): "Animal",
        (192, 0, 128): "Archway",
        (0, 128, 192): "Bicyclist",
        (0, 128, 64): "Bridge",
        (128, 0, 0): "Building",
        (64, 0, 128): "Car",
        (64, 0, 192): "CartLuggagePram",
        (192, 128, 64): "Child",
        (192, 192, 128): "Column_Pole",
        (64, 64, 128): "Fence",
        (128, 0, 192): "LaneMkgsDriv",
        (192, 0, 64): "LaneMkgsNonDriv",
        (128, 128, 64): "Misc_Text",
        (192, 0, 192): "MotorcycleScooter",
        (128, 64, 64): "OtherMoving",
        (64, 192, 128): "ParkingBlock",
        (64, 64, 0): "Pedestrian",
        (128, 64, 128): "Road",
        (128, 128, 192): "RoadShoulder",
        (0, 0, 192): "Sidewalk",
        (192, 128, 128): "SignSymbol",
        (128, 128, 128): "Sky",
        (64, 128, 192): "SUVPickupTruck",
        (0, 0, 64): "TrafficCone",
        (0, 64, 64): "TrafficLight",
        (192, 64, 128): "Train",
        (128, 128, 0): "Tree",
        (192, 128, 192): "Truck_Bus",
        (64, 0, 64): "Tunnel",
        (192, 192, 0): "VegetationMisc",
        (0, 0, 0): "Void",
        (64, 192, 0): "Wall",
    }

    response_channels = len(colors_dict.keys())

    image_size = (256, 256)
    image_size_T = image_size[::-1]
