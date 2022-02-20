#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 30/06/2020
           """

import random
from pathlib import Path
from typing import Tuple, Union

import numpy
import torch
from draugr.numpy_utilities import SplitEnum
from draugr.torch_utilities import SupervisedDataset, global_pin_memory, to_tensor
from matplotlib import pyplot
from torch.utils.data import DataLoader
from warg import drop_unused_kws, passes_kws_to

from neodroidvision.data.classification import DictImageFolder, SplitDictImageFolder

__all__ = ["PairDataset"]


class PairDataset(
    SupervisedDataset
):  # TODO: Extract image specificity of class to a subclass and move this super pair class to a
    # general torch lib.
    """
    # This dataset generates a pair of images. 0 for geniune pair and 1 for imposter pair"""

    @passes_kws_to(DictImageFolder.__init__)
    @drop_unused_kws
    def __init__(
        self,
        data_path: Union[str, Path],
        split: SplitEnum = SplitEnum.training,
        return_categories: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.return_categories = return_categories
        self.split = split
        # name = self.split_names[split]
        if split == SplitEnum.testing:
            self._dataset = DictImageFolder(
                root=data_path / SplitEnum.testing.value, **kwargs
            )
        else:
            self._dataset = SplitDictImageFolder(
                root=data_path / SplitEnum.training.value, split=self.split, **kwargs
            )

    def __getitem__(self, idx1: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        returns torch.tensors for img pair and a label for whether the pair is of the same class (1 if not the
        same)



        :param idx1:
        :type idx1:
        :return:
        :rtype:"""
        t1 = random.choice(self._dataset.category_names)

        if random.randint(0, 1):
            while True:
                t2 = random.choice(self._dataset.category_names)
                if t1 != t2:
                    break
            return (
                self._dataset.sample(t1, idx1)[0],
                self._dataset.sample(
                    t2, random.randint(0, self._dataset.category_sizes[t2])
                )[0],
                torch.ones(1, dtype=torch.long),
                *(t1, t2 if self.return_categories else ()),
            )

        while True:
            idx2 = random.randint(0, self._dataset.category_sizes[t1])
            if idx1 != idx2:
                break

        return (
            self._dataset.sample(t1, idx1)[0],
            self._dataset.sample(t1, idx2)[0],
            torch.zeros(1, dtype=torch.long),
            *(t1, t1 if self.return_categories else ()),
        )

    @property
    def response_shape(self) -> Tuple[int, ...]:
        """

        :return:
        :rtype:"""
        return (len(self._dataset.category_names),)

    @property
    def predictor_shape(self) -> Tuple[int, ...]:
        """

        :return:
        :rtype:"""
        return to_tensor(self.__getitem__(0)[0]).shape

    def __len__(self):
        return len(self._dataset)

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
            images1, images2, *labels = next(dl)
            X1 = numpy.transpose(images1.numpy(), [0, 2, 3, 1])
            X2 = numpy.transpose(images2.numpy(), [0, 2, 3, 1])
            if horizontal_merge:
                X = numpy.dstack((X1, X2))
            else:
                X = numpy.hstack((X1, X2))
            PairDataset.plot_images(X, list(zip(*labels)))

    @staticmethod
    def plot_images(images, label=None) -> None:
        """

        :param images:
        :type images:
        :param label:
        :type label:"""
        images = images.squeeze()
        if label:
            assert len(images) == len(label) == 9, f"{len(images), len(label)}"

        fig, axes = pyplot.subplots(3, 3)
        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i], cmap="Greys_r")

            if label:
                ax.set_xlabel(f"{label[i]}")
            ax.set_xticks([])
            ax.set_yticks([])

        pyplot.show()


if __name__ == "__main__":
    sd = PairDataset(
        Path.home() / "Data" / "mnist_png",
        split=SplitEnum.validation,
        return_categories=True,
    )
    print(sd.predictor_shape)
    print(sd.response_shape)
    sd.sample()
