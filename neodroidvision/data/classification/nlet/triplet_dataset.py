#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 30/06/2020
           """

import random
from pathlib import Path
from typing import Tuple

import numpy
import torch

__all__ = ["TripletDataset"]

from draugr.torch_utilities import global_pin_memory
from torch.utils.data import DataLoader

from neodroidvision.data.classification.nlet import PairDataset


class TripletDataset(
    PairDataset
):  # TODO: Extract image specificity of class to a subclass and move this super pair class to a
    # general torch lib.
    """
    # This dataset generates a triple of images. an image of a category, another of the same category and
    lastly one from another category"""

    def __getitem__(self, idx1: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        returns torch.tensors for img triplet, first tensor being idx random category, second being the same
        category with different index
        and third being of a random other category(Never the same)



        :param idx1:
        :type idx1:
        :return:
        :rtype:"""
        t1 = random.choice(self._dataset.category_names)

        while True:
            idx2 = random.randint(0, self._dataset.category_sizes[t1])
            if idx1 != idx2:
                break

        while True:
            t2 = random.choice(self._dataset.category_names)
            if t1 != t2:
                break

        return (
            self._dataset.sample(t1, idx1)[0],
            self._dataset.sample(t1, idx2)[0],
            self._dataset.sample(
                t2, random.randint(0, self._dataset.category_sizes[t2])
            )[0],
            *(t1, t2 if self.return_categories else ()),
        )

    def sample(self, horizontal_merge: bool = False) -> None:
        """ """
        dl = iter(
            DataLoader(
                self,
                batch_size=9,
                shuffle=True,
                num_workers=0,
                pin_memory=global_pin_memory(0),
            )
        )
        for _ in range(3):
            images1, images2, images3, *labels = next(dl)
            X1 = numpy.transpose(images1.numpy(), [0, 2, 3, 1])
            X2 = numpy.transpose(images2.numpy(), [0, 2, 3, 1])
            X3 = numpy.transpose(images3.numpy(), [0, 2, 3, 1])
            if horizontal_merge:
                X = numpy.dstack((X1, X2, X3))
            else:
                X = numpy.hstack((X1, X2, X3))
            PairDataset.plot_images(X, list(zip(*labels)))


if __name__ == "__main__":
    sd = TripletDataset(Path.home() / "Data" / "mnist_png", return_categories=True)
    print(sd.predictor_shape)
    print(sd.response_shape)
    sd.sample()
