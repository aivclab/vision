from pathlib import Path

import numpy as np
import torch.utils.data
from PIL import Image

from draugr.opencv_utilities import xywh_to_minmax

__all__ = ["COCODataset"]

from neodroidvision.data.datasets.supervised.splitting import Split


class COCODataset(torch.utils.data.Dataset):
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

    data_dirs = {
        "coco_2014_valminusminival": "val2014",
        "coco_2014_minival": "val2014",
        "coco_2014_train": "train2014",
        "coco_2014_val": "val2014",
    }

    splits = {
        "coco_2014_valminusminival": "annotations/instances_valminusminival2014.json",
        "coco_2014_minival": "annotations/instances_minival2014.json",
        "coco_2014_train": "annotations/instances_train2014.json",
        "coco_2014_val": "annotations/instances_val2014.json",
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
    :param remove_empty:
    :type remove_empty:
    """
        from pycocotools.coco import COCO

        self._coco_source = COCO(str(data_root / self.splits[dataset_name]))
        self._data_dir = data_root / self.data_dirs[dataset_name]
        self._img_transform = transform
        self._target_transform = target_transform
        self._remove_empty = split == Split.Training
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
        if self._img_transform:
            image, boxes, labels = self._img_transform(image, boxes, labels)
        if self._target_transform:
            boxes, labels = self._target_transform(boxes, labels)
        targets = dict(boxes=boxes, labels=labels)
        return image, targets, index

    def get_annotation(self, index):
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
        boxes = np.array(
            [xywh_to_minmax(obj["bbox"]) for obj in ann], np.float32
        ).reshape((-1, 4))
        labels = np.array(
            [self._coco_id_to_contiguous_id[obj["category_id"]] for obj in ann],
            np.int64,
        ).reshape((-1,))

        keep = (boxes[:, 3] > boxes[:, 1]) & (
            boxes[:, 2] > boxes[:, 0]
        )  # remove invalid boxes
        return boxes[keep], labels[keep]

    def get_img_info(self, index):
        image_id = self._ids[index]
        img_data = self._coco_source.imgs[image_id]
        return img_data

    def _read_image(self, image_id):
        file_name = self._coco_source.loadImgs(image_id)[0]["file_name"]
        image_file = self._data_dir / file_name
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image
