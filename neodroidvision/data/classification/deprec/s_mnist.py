#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 25/03/2020
           """

from pathlib import Path
from typing import Tuple

import torch
from torch.utils import data
from torch.utils.data import Subset
from torchvision import transforms

__all__ = ["MNISTDataset"]

from torchvision.datasets import MNIST

from draugr.torch_utilities import Split, SplitByPercentage, SupervisedDataset


class MNISTDataset(SupervisedDataset):
    """

"""

    @property
    def response_shape(self) -> Tuple[int, ...]:
        """

:return:
:rtype:
"""
        return (len(self.categories),)

    @property
    def predictor_shape(self) -> Tuple[int, ...]:
        """

:return:
:rtype:
"""
        return self._resize_shape

    def __init__(
        self,
        dataset_path: Path,
        split: Split = Split.Training,
        validation: float = 0.3,
        resize_s: int = 28,
        seed: int = 42,
        download=True,
    ):
        """
:param dataset_path: dataset directory
:param split: train, valid, test
"""
        super().__init__()

        if not download:
            assert dataset_path.exists(), f"root: {dataset_path} not found."

        self._resize_shape = (1, resize_s, resize_s)

        train_trans = transforms.Compose(
            [
                transforms.RandomResizedCrop(resize_s),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        val_trans = transforms.Compose(
            [
                transforms.Resize(resize_s),
                # transforms.CenterCrop(resize_s),
                transforms.ToTensor(),
            ]
        )

        if split == Split.Training:
            mnist_data = MNIST(
                str(dataset_path), train=True, download=download, transform=train_trans
            )
        elif split == Split.Validation:
            mnist_data = MNIST(
                str(dataset_path), train=True, download=download, transform=val_trans
            )
        else:
            mnist_data = MNIST(
                str(dataset_path), train=False, download=download, transform=val_trans
            )

        if split != Split.Testing:
            torch.manual_seed(seed)
            train_ind, val_ind, test_ind = SplitByPercentage(
                len(mnist_data), validation=validation, testing=0.0
            ).shuffled_indices()
            if split == Split.Validation:
                self.mnist_data_split = Subset(mnist_data, val_ind)
            else:
                self.mnist_data_split = Subset(mnist_data, train_ind)
        else:
            self.mnist_data_split = mnist_data

        self.categories = mnist_data.classes

    def __len__(self):
        return len(self.mnist_data_split)

    def __getitem__(self, index):
        return self.mnist_data_split.__getitem__(index)


if __name__ == "__main__":

    def siuadyh():
        import tqdm

        batch_size = 32

        dt_t = MNISTDataset(Path(Path.home() / "Data" / "mnist"), split=Split.Training)

        print(len(dt_t))

        dt_v = MNISTDataset(
            Path(Path.home() / "Data" / "mnist"), split=Split.Validation
        )

        print(len(dt_v))

        dt = MNISTDataset(Path(Path.home() / "Data" / "mnist"), split=Split.Testing)

        print(len(dt))

        data_loader = torch.utils.data.DataLoader(
            dt, batch_size=batch_size, shuffle=False
        )

        for batch_idx, (imgs, label) in tqdm.tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc="Bro",
            ncols=80,
            leave=False,
        ):
            # pyplot.imshow(dt.inverse_transform(imgs[0]))
            # pyplot.imshow(imgs)
            # pyplot.show()
            print(imgs.shape)
            break

    siuadyh()
