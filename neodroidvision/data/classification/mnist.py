#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 25/03/2020
           """

from pathlib import Path
from typing import Sequence, Tuple

import numpy
import torch
from draugr.numpy_utilities import SplitEnum
from draugr.torch_utilities import SupervisedDataset, global_pin_memory
from matplotlib import pyplot
from torch.utils.data import Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

__all__ = ["MNISTDataset"]


class MNISTDataset(SupervisedDataset):
    """"""

    def __init__(self, data_dir: Path, split: SplitEnum = SplitEnum.training):
        super().__init__()
        if split == SplitEnum.training:
            self._dataset = datasets.MNIST(
                str(data_dir), train=True, download=True, transform=self.trans
            )
        else:
            self._dataset = datasets.MNIST(
                str(data_dir), train=False, download=True, transform=self.trans
            )

    def __getitem__(self, index):
        return self._dataset.__getitem__(index)

    def __len__(self):
        return self._dataset.__len__()

    @property
    def predictor_shape(self) -> Tuple[int, ...]:
        """

        :return:
        :rtype:"""
        return 1, 28, 28

    @property
    def response_shape(self) -> Tuple[int, ...]:
        """

        :return:
        :rtype:"""
        return (10,)

    trans = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    inverse_transform = transforms.Compose(
        [
            # transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist()), # TODO: imp
            transforms.ToPILImage()
        ]
    )

    @staticmethod
    def get_train_valid_loader(
        data_dir: Path,
        *,
        batch_size: int,
        random_seed: int,
        valid_size: float = 0.1,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        using_cuda: bool = True,
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Train and validation data loaders.

        If using CUDA, num_workers should be set to 1 and pin_memory to True.

        Args:
        data_dir: path directory to the dataset.
        batch_size: how many samples per batch to load.
        random_seed: fix seed for reproducibility.
        valid_size: percentage split of the training set used for
            the validation set. Should be a float in the range [0, 1].
            In the paper, this number is set to 0.1.
        shuffle: whether to shuffle the train/validation indices.
        show_sample: plot 9x9 sample grid of the dataset.
        num_workers: number of subprocesses to use when loading the dataset.
        pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
            True if using GPU.
            :param data_dir:
            :type data_dir:
            :param batch_size:
            :type batch_size:
            :param random_seed:
            :type random_seed:
            :param valid_size:
            :type valid_size:
            :param shuffle:
            :type shuffle:
            :param num_workers:
            :type num_workers:
            :param pin_memory:
            :type pin_memory:
            :param using_cuda:
            :type using_cuda:"""
        error_msg = "[!] valid_size should be in the range [0, 1]."
        assert (valid_size >= 0) and (valid_size <= 1), error_msg

        if using_cuda:
            pass
            # assert num_workers == 1
            # assert pin_memory == True

        dataset = MNISTDataset(data_dir)
        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(numpy.floor(valid_size * num_train))

        if shuffle:
            numpy.random.seed(random_seed)
            numpy.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        valid_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=valid_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return train_loader, valid_loader

    @staticmethod
    def get_test_loader(
        data_dir: Path,
        batch_size: int,
        *,
        num_workers: int = 0,
        pin_memory: bool = False,
        using_cuda: bool = True,
    ) -> torch.utils.data.DataLoader:
        """Test datalaoder.

        If using CUDA, num_workers should be set to 1 and pin_memory to True.

        Args:
        data_dir: path directory to the dataset.
        batch_size: how many samples per batch to load.
        num_workers: number of subprocesses to use when loading the dataset.
        pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
            True if using GPU.
        :param data_dir:
        :type data_dir:
        :param batch_size:
        :type batch_size:
        :param num_workers:
        :type num_workers:
        :param pin_memory:
        :type pin_memory:
        :param using_cuda:
        :type using_cuda:"""
        # define transforms

        if using_cuda:
            pass
            # assert num_workers == 1
            # assert pin_memory == True

        dataset = MNISTDataset(data_dir, split=SplitEnum.testing)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return data_loader

    def sample(self) -> None:
        """"""
        images, labels = next(
            iter(
                torch.utils.data.DataLoader(
                    self,
                    batch_size=9,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=global_pin_memory(0),
                )
            )
        )
        X = images.numpy()
        X = numpy.transpose(X, [0, 2, 3, 1])
        MNISTDataset.plot_images(X, labels)

    @staticmethod
    def plot_images(images: numpy.ndarray, label: Sequence) -> None:
        """

        :param images:
        :type images:
        :param label:
        :type label:"""
        images = images.squeeze()
        assert len(images) == len(label) == 9

        fig, axes = pyplot.subplots(3, 3)
        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i], cmap="Greys_r")
            xlabel = f"{label[i]}"
            ax.set_xlabel(xlabel)
            ax.set_xticks([])
            ax.set_yticks([])

        pyplot.show()


if __name__ == "__main__":

    def a():
        """ """
        MNISTDataset(Path.home() / "Data" / "MNIST").sample()
        pyplot.show()

    a()
