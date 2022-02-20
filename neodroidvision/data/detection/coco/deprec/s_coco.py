#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/03/2020
           """

from pathlib import Path
from typing import Tuple

import numpy
from PIL import Image
from draugr.numpy_utilities import SplitEnum
from draugr.opencv_utilities import xywh_to_minmax

from neodroidvision.data.detection.object_detection_dataset import (
    ObjectDetectionDataset,
)

__all__ = ["COCODataset"]

from draugr.torch_utilities import NamedTensorTuple


class COCODataset(ObjectDetectionDataset):
    """ """

    @property
    def response_shape(self) -> Tuple[int, ...]:
        """

        Args:
          self:
        """
        raise NotImplementedError

    categories = (
        "__background__",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    )

    image_dirs = {
        "coco_2014_valminusminival": "images/val2014",
        "coco_2014_minival": "images/val2014",
        "coco_2014_train": "images/train2014",
        "coco_2014_val": "images/val2014",
    }

    splits = {
        "coco_2014_valminusminival": "annotations/instances_valminusminival2014.json",
        "coco_2014_minival": "annotations/instances_minival2014.json",
        "coco_2014_train": "annotations/instances_train2014.json",
        "coco_2014_val": "annotations/instances_val2014.json",
    }

    @property
    def predictor_shape(self) -> Tuple[int, ...]:
        """ """
        raise NotImplementedError

    def __init__(
        self,
        data_root: Path,
        dataset_name: str,
        split: SplitEnum,
        img_transform: callable = None,
        annotation_transform: callable = None,
    ):
        """

        :param data_root:
        :type data_root:
        :param dataset_name:
        :type dataset_name:
        :param split:
        :type split:
        :param img_transform:
        :type img_transform:
        :param annotation_transform:
        :type annotation_transform:
        :param remove_empty:
        :type remove_empty:"""
        super().__init__(
            data_root, dataset_name, split, img_transform, annotation_transform
        )
        from pycocotools.coco import COCO

        self._coco_source = COCO(str(data_root / self.splits[dataset_name]))
        self._image_dir = data_root / self.image_dirs[dataset_name]
        self._img_transforms = img_transform
        self._annotation_transforms = annotation_transform
        self._remove_empty = split == SplitEnum.training
        if self._remove_empty:
            self._ids = list(
                self._coco_source.imgToAnns.keys()
            )  # when training, images without annotations are removed.
        else:
            self._ids = list(
                self._coco_source.imgs.keys()
            )  # when testing, all images used.
        self._coco_id_to_contiguous_id = {
            coco_id: i + 1
            for i, coco_id in enumerate(sorted(self._coco_source.getCatIds()))
        }
        self._contiguous_id_to_coco_id = {
            v: k for k, v in self._coco_id_to_contiguous_id.items()
        }

    def __getitem__(self, index):
        image_id = self._ids[index]
        boxes, labels = self._get_annotation(image_id)
        image = self._read_image(image_id)
        if self._img_transforms:
            image, boxes, labels = self._img_transforms(image, boxes, labels)
        if self._annotation_transforms:
            boxes, labels = self._annotation_transforms(boxes, labels)
        return image, NamedTensorTuple(boxes=boxes, labels=labels), index

    def get_annotation(self, index):
        """

        :param index:
        :type index:
        :return:
        :rtype:"""
        image_id = self._ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self._ids)

    def _get_annotation(self, image_id):
        ann = [
            obj
            for obj in self._coco_source.loadAnns(
                self._coco_source.getAnnIds(imgIds=image_id)
            )
            if obj["iscrowd"] == 0
        ]  # filter crowd annotations
        boxes = numpy.array(
            [xywh_to_minmax(obj["bbox"]) for obj in ann], numpy.float32
        ).reshape((-1, 4))
        labels = numpy.array(
            [self._coco_id_to_contiguous_id[obj["category_id"]] for obj in ann],
            numpy.int64,
        ).reshape((-1,))

        keep = (boxes[:, 3] > boxes[:, 1]) & (
            boxes[:, 2] > boxes[:, 0]
        )  # remove invalid boxes
        return boxes[keep], labels[keep]

    def get_img_info(self, index):
        """

        :param index:
        :type index:
        :return:
        :rtype:"""
        return self._coco_source.imgs[self._ids[index]]

    def _read_image(self, image_id):
        return numpy.array(
            Image.open(
                self._image_dir / self._coco_source.loadImgs(image_id)[0]["file_name"]
            ).convert("RGB")
        )
