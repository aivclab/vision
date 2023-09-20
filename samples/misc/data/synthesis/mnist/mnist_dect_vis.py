#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
           """

import numpy
import pathlib
from PIL import Image
from matplotlib import pyplot
from neodroidvision import PROJECT_APP_PATH
from numpy import ndarray
from typing import Tuple


def read_labels(label_path: pathlib.Path) -> Tuple[ndarray, ndarray]:
    """

    :param label_path:
    :type label_path:
    :return:
    :rtype:
    """
    assert label_path.is_file()
    labels = []
    BBOXES_XYXY = []

    with open(label_path, "r") as fp:
        for line in list(fp.readlines())[1:]:
            label, xmin, ymin, xmax, ymax = [int(_) for _ in line.split(",")]
            labels.append(label)
            BBOXES_XYXY.append([xmin, ymin, xmax, ymax])

    return numpy.array(labels), numpy.array(BBOXES_XYXY)


if __name__ == "__main__":
    from draugr.opencv_utilities import draw_bounding_boxes

    base_path = pathlib.Path(
        PROJECT_APP_PATH.user_data / "Data" / "mnist_detection" / "train"
    )
    image_dir = base_path / "images"
    label_dir = base_path / "annotations"
    for impath in image_dir.glob("*.png"):
        labels, bboxes_XYXY = read_labels(
            (label_dir / f"{impath.stem}").with_suffix(".csv")
        )
        im = Image.open(str(impath))  # .convert('RGB')

        pyplot.imshow(draw_bounding_boxes(im, bboxes_XYXY, labels=labels))
        pyplot.show()
    else:
        print("Found no images")
