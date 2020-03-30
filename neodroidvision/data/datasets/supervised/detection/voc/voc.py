import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch.utils.data
from PIL import Image

__all__ = ["VOCDataset"]

from neodroidvision.data.datasets.supervised.splitting import Split


class VOCDataset(torch.utils.data.Dataset):
    categories = (
        "__background__",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )

    data_dirs = {
        "voc_2007_train": "VOC2007",
        "voc_2007_val": "VOC2007",
        "voc_2007_trainval": "VOC2007",
        "voc_2007_test": "VOC2007",
        "voc_2012_train": "VOC2012",
        "voc_2012_val": "VOC2012",
        "voc_2012_trainval": "VOC2012",
        "voc_2012_test": "VOC2012",
    }

    splits = {
        "voc_2007_train": "train",
        "voc_2007_val": "val",
        "voc_2007_trainval": "trainval",
        "voc_2007_test": "test",
        "voc_2012_train": "train",
        "voc_2012_val": "val",
        "voc_2012_trainval": "trainval",
        "voc_2012_test": "test",
    }

    def __init__(
        self,
        data_root: Path,
        dataset_name: str,
        split: Split,
        transform: callable = None,
        target_transform: callable = None,
    ):
        """

    Dataset for VOC data.

    data_root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following

    Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.

    :param data_root:
    :type data_root:
    :param dataset_name:
    :type dataset_name:
    :param split:
    :type split:
    :param transform:
    :type transform:
    :param target_transform:
    :type target_transform:
    :param keep_difficult:
    :type keep_difficult:
    """

        self._data_dir = data_root / self.data_dirs[dataset_name]
        self._img_transforms = transform
        self._target_transform = target_transform

        self._ids = VOCDataset._read_image_ids(
            self._data_dir / "ImageSets" / "Main" / f"{self.splits[dataset_name]}.txt"
        )
        self._keep_difficult = not split == Split.Training

        self._class_dict = {
            class_name: i for i, class_name in enumerate(self.categories)
        }

    def __getitem__(self, index):
        image_id = self._ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self._keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        if self._img_transforms:
            image, boxes, labels = self._img_transforms(image, boxes, labels)
        if self._target_transform:
            boxes, labels = self._target_transform(boxes, labels)
        targets = dict(boxes=boxes, labels=labels)
        return image, targets, index

    def get_annotation(self, index):
        image_id = self._ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self._ids)

    @staticmethod
    def _read_image_ids(image_sets_file: Path):
        ids = []
        with open(str(image_sets_file)) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_file = self._data_dir / "Annotations" / f"{image_id}.xml"

        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for obj in objects:
            class_name = obj.find("name").text.lower().strip()
            bbox = obj.find("bndbox")
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find("xmin").text) - 1
            y1 = float(bbox.find("ymin").text) - 1
            x2 = float(bbox.find("xmax").text) - 1
            y2 = float(bbox.find("ymax").text) - 1
            boxes.append([x1, y1, x2, y2])
            labels.append(self._class_dict[class_name])
            is_difficult_str = obj.find("difficult").text
            is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (
            np.array(boxes, dtype=np.float32),
            np.array(labels, dtype=np.int64),
            np.array(is_difficult, dtype=np.uint8),
        )

    def get_img_info(self, index):
        img_id = self._ids[index]
        annotation_file = self._data_dir / "Annotations" / f"{img_id}.xml"
        anno = ET.parse(annotation_file).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def _read_image(self, image_id):
        image_file = self._data_dir / "JPEGImages" / f"{image_id}.jpg"
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image
