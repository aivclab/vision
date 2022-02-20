#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/03/2020
           """

from pathlib import Path

from draugr.tqdm_utilities import progress_bar
from warg import Triple

"""Get the binarized MNIST dataset and convert to hdf5.
From https://github.com/yburda/iwae/blob/master/datasets.py
"""
import urllib.request

import h5py
import numpy
from neodroidvision import PROJECT_APP_PATH


def parse_binary_mnist(data_dir: Path) -> Triple:
    """

    Args:
      data_dir:

    Returns:

    """

    def lines_to_np_array(lines):
        """

        Args:
          lines:

        Returns:

        """
        return numpy.array([[int(i) for i in line.split()] for line in lines])

    with open(str(data_dir / "binarized_mnist_train.amat")) as f:
        lines = f.readlines()
    train_data = lines_to_np_array(lines).astype("float32")
    with open(str(data_dir / "binarized_mnist_valid.amat")) as f:
        lines = f.readlines()
    validation_data = lines_to_np_array(lines).astype("float32")
    with open(str(data_dir / "binarized_mnist_test.amat")) as f:
        lines = f.readlines()
    test_data = lines_to_np_array(lines).astype("float32")
    return train_data, validation_data, test_data


def download_binary_mnist(
    file_path: str = "binary_mnist.h5",
    data_dir: Path = (PROJECT_APP_PATH.user_data / "vanilla_vae" / "data"),
):
    """

    Args:
      file_path:
      data_dir:
    """
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
    subdatasets = ["train", "valid", "test"]
    for subdataset in progress_bar(subdatasets):
        filename = f"binarized_mnist_{subdataset}.amat"
        url = (
            f"http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist"
            f"/binarized_mnist_{subdataset}.amat"
        )
        local_filename = str(data_dir / filename)
        urllib.request.urlretrieve(url, local_filename)

    train, validation, test = parse_binary_mnist(data_dir)

    data_dict = {"train": train, "valid": validation, "test": test}
    f = h5py.File(file_path, "w")
    f.create_dataset("train", data=data_dict["train"])
    f.create_dataset("valid", data=data_dict["valid"])
    f.create_dataset("test", data=data_dict["test"])
    f.close()
    print(f"Saved binary MNIST data to: {file_path}")


if __name__ == "__main__":
    download_binary_mnist()
