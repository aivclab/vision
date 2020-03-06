#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 01/03/2020
           """

from enum import Enum
from typing import Tuple

import numpy
from torch.utils.data import Dataset

__all__ = ["Split", "SplitByPercentage", "SupervisedDataset"]


class Split(Enum):
    Training = 0
    Validation = 1
    Testing = 2


class SplitByPercentage:
    default_split_names = ("training", "validation", "testing")

    def __init__(self, dataset_length: int, training=0.7, validation=0.2, testing=0.1):
        self.total_num = dataset_length
        split = numpy.array([training, validation, testing])
        self.normalised = split / sum(split)
        self.training_percentage, self.validation_percentage, self.testing_percentage = (
            self.normalised
        )
        self.training_num, self.validation_num, self.testing_num = self.unnormalised(
            dataset_length
        )

    def unnormalised(self, num: int, floored: bool = True) -> numpy.ndarray:
        unnorm = self.normalised * num
        if floored:
            unnorm = numpy.floor(unnorm)
        return unnorm.astype(int)

    def __repr__(self) -> str:
        return str({k: n for k, n in zip(self.default_split_names, self.normalised)})


class SupervisedDataset(Dataset):
    @property
    def split_names(self) -> Tuple[str, str, str]:
        return SplitByPercentage.default_split_names


if __name__ == "__main__":
    print(SplitByPercentage(521))
    print(SplitByPercentage(2512).unnormalised(123))
