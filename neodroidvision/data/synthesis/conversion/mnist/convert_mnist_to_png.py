#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian"
__doc__ = r"""

           Created on 22/03/2020
           """

__all__ = ["convert_to_mnist_png", "generate_mnist_png"]

import os
import struct
import sys
from array import array
from pathlib import Path
from typing import Tuple

import torchvision.datasets
from draugr.numpy_utilities import SplitEnum

from neodroidvision import PROJECT_APP_PATH


def read_data(
    dataset: SplitEnum = SplitEnum.training,
    path: Path = PROJECT_APP_PATH.user_cache / "mnist",
) -> Tuple:
    """

    Args:
      dataset:
      path:

    Returns:

    """
    if dataset is SplitEnum.training:
        file_name_img = path / "train-images-idx3-ubyte"
        file_name_category = path / "train-labels-idx1-ubyte"
    elif dataset is SplitEnum.testing:
        file_name_img = path / "t10k-images-idx3-ubyte"
        file_name_category = path / "t10k-labels-idx1-ubyte"
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    with open(file_name_category, "rb") as category_file:
        magic_nr, size = struct.unpack(">II", category_file.read(8))
        category = array("b", category_file.read())

    with open(file_name_img, "rb") as image_file:
        magic_nr, size, rows, cols = struct.unpack(">IIII", image_file.read(16))
        image = array("B", image_file.read())

    return category, image, size, rows, cols


def write_dataset(labels, data, size, rows, cols, output_dir) -> None:
    """

    Args:
        labels:
        data:
        size:
        rows:
        cols:
        output_dir:

    Returns:

    """
    output_dirs = [output_dir / str(i) for i in range(10)]
    for dir in output_dirs:  # create output directories
        if not dir.exists():
            os.makedirs(dir)
    import png  # pip install pypng

    for (i, label) in enumerate(labels):
        output_filename = output_dirs[label] / f"{str(i)}.png"
        print(f"writing {output_filename}")
        with open(output_filename, "wb") as h:
            w = png.Writer(cols, rows, greyscale=True)
            data_i = [
                data[(i * rows * cols + j * cols) : (i * rows * cols + (j + 1) * cols)]
                for j in range(rows)
            ]
            w.write(h, data_i)


def convert_to_mnist_png(input_path, output_path) -> None:
    """

    Args:
      input_path:
      output_path:
    """
    for dataset in [SplitEnum.training, SplitEnum.testing]:
        write_dataset(*read_data(dataset, input_path), output_path / dataset.value)


def generate_mnist_png(path=PROJECT_APP_PATH.user_cache) -> None:
    """

    Args:
      path:
    """
    base = path / "mnist"
    torchvision.datasets.MNIST(str(base), download=True)
    torchvision.datasets.MNIST(str(base), train=False, download=True)
    convert_to_mnist_png(base / "MNIST" / "raw", path / "mnist_png")
    print(f'generated mnist_png at {path / "mnist_png"}')


if __name__ == "__main__":

    def main2():
        """ """
        if len(sys.argv) != 3:
            print(f"usage: {sys.argv[0]} <input_path> <output_path>")
            sys.exit()

        input_path = sys.argv[1]
        output_path = sys.argv[2]

        convert_to_mnist_png(input_path, output_path)

    generate_mnist_png()
