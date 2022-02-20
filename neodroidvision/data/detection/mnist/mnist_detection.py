#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "{user}"
__doc__ = r"""

           Created on {date}
           """

from typing import Tuple

from neodroidvision.data.detection.object_detection_dataset import (
    ObjectDetectionDataset,
)


class MnistDetectionDataset(ObjectDetectionDataset):
    def __init__(self, download: bool = True):
        super().__init__()
        # TODO: FINISH

    @property
    def response_shape(self) -> Tuple[int, ...]:
        pass

    @property
    def predictor_shape(self) -> Tuple[int, ...]:
        pass

    def __len__(self):
        pass

    def __getitem__(self, index: int):
        pass
