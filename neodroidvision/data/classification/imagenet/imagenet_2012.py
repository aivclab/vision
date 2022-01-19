#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 27/06/2020
           """

from pathlib import Path
from typing import Dict, Tuple

import numpy
import torch
import torchvision
from draugr.numpy_utilities import SplitEnum
from draugr.torch_utilities import SupervisedDataset
from matplotlib import pyplot
from torch.utils import data
from torchvision import transforms

from neodroidvision.data.classification.imagenet.imagenet_2012_id import categories_id
from neodroidvision.data.classification.imagenet.imagenet_2012_names import (
    categories_names,
)

__all__ = ["ImageNet2012"]


class ImageNet2012(SupervisedDataset):
    """ """

    mean = numpy.array([0.485, 0.456, 0.406])
    std = numpy.array([0.229, 0.224, 0.225])

    category_names = categories_names
    category_id = categories_id

    inverse_base_transform = transforms.Compose(
        [
            transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist()),
            transforms.ToPILImage(),
        ]
    )

    base_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    @property
    def response_shape(self) -> Tuple[int, ...]:
        """

        :return:
        :rtype:"""
        return (1000,)

    @property
    def predictor_shape(self) -> Tuple[int, ...]:
        """

        :return:
        :rtype:"""
        return self._crop_size, self._crop_size

    @property
    def split_names(self) -> Dict[SplitEnum, str]:
        """

        :return:
        :rtype:"""
        return {
            SplitEnum.training: "train",
            SplitEnum.validation: "val",
            SplitEnum.testing: "test",
        }

    def __init__(
        self,
        dataset_path: Path,
        split: SplitEnum = SplitEnum.training,
        resize_s: int = 256,
        crop_size: int = 224,
    ):
        """
        :type resize_s: int or tuple(w,h)
        :param dataset_path: dataset directory
        :param split: train, valid, test"""
        super().__init__()
        if isinstance(dataset_path, str):
            dataset_path = Path(dataset_path)
        assert dataset_path.exists(), f"root: {dataset_path} not found."
        assert resize_s > 2, "resize_s should be >2"
        assert crop_size > 2, "crop_size should be >2"

        self._crop_size = crop_size

        self._split = split
        self._dataset_path = dataset_path / self.split_names[split]

        self.train_trans = transforms.Compose(
            [
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                self.base_transform,
            ]
        )

        self.val_trans = transforms.Compose(
            [
                transforms.Resize(resize_s),
                transforms.CenterCrop(crop_size),
                self.base_transform,
            ]
        )

        self._image_folder = torchvision.datasets.ImageFolder(
            str(self._dataset_path), self.val_trans
        )

    def __len__(self) -> int:
        return len(self._image_folder)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        return image and category

        :param index:
        :type index:
        :return:
        :rtype:"""
        return self._image_folder[index]


if __name__ == "__main__":

    def main():
        """ """
        import tqdm

        batch_size = 32

        dt = ImageNet2012(
            Path.home() / "Data" / "Datasets" / "ILSVRC2012", split=SplitEnum.validation
        )

        val_loader = torch.utils.data.DataLoader(
            dt, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
        )

        for batch_idx, (imgs, categories) in tqdm.tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            desc="Bro",
            ncols=80,
            leave=False,
        ):
            pyplot.imshow(dt.inverse_base_transform(imgs[0]))
            pyplot.title(dt.category_names[categories[0].item()])
            pyplot.show()
            break

    main()
