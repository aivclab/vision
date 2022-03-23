#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian"
__doc__ = r"""

           Created on 29/03/2020
           """

import gzip
import pickle

import numpy
from draugr.numpy_utilities import SplitEnum
from tqdm import trange

from augmentation import rotate_y
from gen import make_voxel, img_to_point_cloud
from neodroidvision import PROJECT_APP_PATH

__all__ = ["save_dataset"]


def save_dataset(X, y, voxel, output, shape=(28, 28)):
    """

    Args:
      X:
      y:
      voxel:
      output:
      shape:
    """
    img = numpy.zeros((shape[0] + 2, shape[1] + 2))
    import nrrd  # pip install pynrrd

    for i in trange(len(X)):
        img[1:-1, 1:-1] = X[i].reshape(shape[0], shape[1])
        data = img_to_point_cloud(img, voxel)

        # rotate to vertical
        transf = numpy.c_[data[:, :3], numpy.ones(data[:, :3].shape[0])]
        transf = transf @ rotate_y(90)
        data[:, :3] = transf[:, :-1]

        if i == 1:
            # data = numpy.zeros((70, 800, 600))
            nrrd.write(
                str(PROJECT_APP_PATH.user_data / "mnist3d" / f"{y[i]}.nrrd"),
                data[:, :3],
            )
            exit(0)


if __name__ == "__main__":

    with gzip.open(PROJECT_APP_PATH.user_data / "mnist.pkl.gz", "rb") as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="iso-8859-1")

        N_VALID = 100
        for split, set_ in zip(
            (SplitEnum.training, SplitEnum.validation, SplitEnum.testing),
            (train_set, valid_set, test_set),
        ):
            save_dataset(
                set_[0][:N_VALID],
                set_[1][:N_VALID],
                make_voxel(),
                PROJECT_APP_PATH.user_data / "mnist3d" / split.value,
            )
