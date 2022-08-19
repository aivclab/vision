#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 28/07/2020
           """

__all__ = ["SplitDictImageFolder", "DictImageFolder"]

from pathlib import Path

from draugr.numpy_utilities import SplitEnum
from draugr.torch_utilities import (
    DictDatasetFolder,
    SplitDictDatasetFolder,
)
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
from torchvision.transforms import transforms


class SplitDictImageFolder(SplitDictDatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

      root/dog/xxx.png
      root/dog/xxy.png
      root/dog/xxz.png

      root/cat/123.png
      root/cat/nsdf3.png
      root/cat/asd932_.png

    Args:
      root (string): Root directory path.
      transform (callable, optional): A function/transform that  takes in an PIL image
          and returns a transformed version. E.g, ``transforms.RandomCrop``
      target_transform (callable, optional): A function/transform that takes in the
          target and transforms it.
      loader (callable, optional): A function to load an image given its path.
      is_valid_file (callable, optional): A function that takes path of an Image file
          and check if the file is a valid file (used to check of corrupt files)

    Attributes:
      classes (list): List of the class names sorted alphabetically.
      imgs (list): List of (image path, class_index) tuples"""

    def __init__(
        self,
        root: Path,
        transform: callable = transforms.ToTensor(),
        target_transform: callable = None,
        loader=default_loader,
        split: SplitEnum = SplitEnum.training,
    ):
        super().__init__(
            root,
            loader,
            extensions=IMG_EXTENSIONS,
            transform=transform,
            target_transform=target_transform,
            split=split,
        )
        self.imgs = self._data_categories


class DictImageFolder(DictDatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

      root/dog/xxx.png
      root/dog/xxy.png
      root/dog/xxz.png

      root/cat/123.png
      root/cat/nsdf3.png
      root/cat/asd932_.png

    Args:
      root (string): Root directory path.
      transform (callable, optional): A function/transform that  takes in an PIL image
          and returns a transformed version. E.g, ``transforms.RandomCrop``
      target_transform (callable, optional): A function/transform that takes in the
          target and transforms it.
      loader (callable, optional): A function to load an image given its path.
      is_valid_file (callable, optional): A function that takes path of an Image file
          and check if the file is a valid file (used to check of corrupt files)

    Attributes:
      classes (list): List of the class names sorted alphabetically.
      imgs (list): List of (image path, class_index) tuples"""

    def __init__(
        self,
        root: Path,
        transform: callable = transforms.ToTensor(),
        target_transform: callable = None,
        loader=default_loader,
        is_valid_file: callable = None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self._data
