#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/03/2020
           """

import copy
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import torch
import torch.utils.data
import torchvision
from numpy.core.multiarray import ndarray
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision.transforms import Compose

__all__ = [
    "FilterAndRemapCocoCategories",
    "convert_coco_poly_to_mask",
    "ConvertCocoPolysToMask",
    "_coco_remove_images_without_annotations",
    "convert_to_coco_api",
    "get_coco_api_from_dataset",
    "CocoDetection",
    "get_coco_ins",
    "get_coco_kp",
    "CocoMask",
    "CocoPolyAnnotation",
    "CocoModeEnum",
]

from draugr.torch_utilities import NamedTensorTuple
from draugr.numpy_utilities import SplitEnum
from warg import NOD


class CocoModeEnum(Enum):
    instances = "instances"
    person_keypoints = "person_keypoints"


CocoPolyAnnotation = NOD(
    {
        "image_id": None,
        "bbox": None,
        "category_id": None,
        "area": None,
        "iscrowd": None,
        "id": None,
        "segmentation": None,
        "keypoints": None,
        "num_keypoints": None,
    }
)

CocoMask = NOD(
    {
        "boxes": None,
        "labels": None,
        "masks": None,
        "image_id": None,
        "area": None,
        "iscrowd": None,
        "keypoints": None,
    }
)


class FilterAndRemapCocoCategories(object):
    """ """

    def __init__(self, categories: List[str], remap: bool = True):
        self._categories = categories
        self._remap = remap

    def __call__(self, image, target: Dict[str, Any]) -> Tuple:
        anno = target["annotations"]
        anno = [obj for obj in anno if obj["category_id"] in self._categories]
        if not self._remap:
            target["annotations"] = anno
            return image, target
        anno = copy.deepcopy(anno)
        for obj in anno:
            obj["category_id"] = self._categories.index(obj["category_id"])
        target["annotations"] = anno
        return image, target


def convert_coco_poly_to_mask(
    segmentations: Sequence, height: int, width: int
) -> NamedTensorTuple:
    """

    :param segmentations:
    :type segmentations:
    :param height:
    :type height:
    :param width:
    :type width:
    :return:
    :rtype:"""
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return NamedTensorTuple(masks=masks)


class ConvertCocoPolysToMask(object):
    def __call__(self, image: ndarray, target: Mapping[str, Any]) -> Tuple:
        w, h = image.size

        image_id = torch.tensor([target["image_id"]])

        anno = [obj for obj in target["annotations"] if obj.iscrowd == 0]
        boxes = [obj.BoundingBox for obj in anno]

        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = torch.tensor([obj.category_id for obj in anno], dtype=torch.int64)

        masks = convert_coco_poly_to_mask([obj.segmentation for obj in anno], h, w)

        keypoints = None
        if anno and anno[0].Keypoints is not None:
            keypoints = [obj.Keypoints for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

        target = NOD(
            boxes=boxes[keep],
            labels=classes[keep],
            masks=masks[keep],
            image_id=image_id,
            area=torch.tensor([obj.area for obj in anno]),
            iscrowd=torch.tensor([obj.iscrowd for obj in anno]),
            keypoints=None,
        )

        if keypoints is not None:
            target.keypoints = keypoints[keep]

        return image, target


def _coco_remove_images_without_annotations(
    dataset: Dataset,
    category_list: Sequence[CocoPolyAnnotation] = None,
    min_keypoints_per_image: int = 10,
) -> Dataset:
    def _has_only_empty_bbox(anno: List[CocoPolyAnnotation]) -> bool:
        return all(any(o <= 1 for o in obj.BoundingBox[2:]) for obj in anno)

    def _count_visible_keypoints(anno: List[CocoPolyAnnotation]) -> int:
        return sum(sum(1 for v in ann.Keypoints[2::3] if v > 0) for ann in anno)

    def _has_valid_annotation(anno: List[CocoPolyAnnotation]) -> bool:
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False
        # keypoints task have a slight different critera for considering
        # if an annotation is valid
        if anno[0].Keypoints is None:
            return True
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        if _count_visible_keypoints(anno) >= min_keypoints_per_image:
            return True
        return False

    assert isinstance(dataset, torchvision.datasets.CocoDetection)
    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if category_list:
            anno = [obj for obj in anno if obj.category_id in category_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


def convert_to_coco_api(ds):
    """

    :param ds:
    :type ds:
    :return:
    :rtype:"""
    coco_ds = COCO()
    ann_id = 0
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"].item()
        dataset["images"].append(
            {"id": image_id, "height": img.shape[-2], "width": img.shape[-1]}
        )
        bboxes = targets["boxes"]
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        masks = None
        keypoints = None
        if "masks" in targets:
            masks = targets["masks"]
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = NOD(
                image_id=image_id,
                bbox=bboxes[i],
                category_id=labels[i],
                area=areas[i],
                iscrowd=iscrowd[i],
                id=ann_id,
                segmentation=None,
                keypoints=None,
                num_keypoints=None,
            )
            categories.add(labels[i])
            if "masks" in targets:
                ann.segmentation = coco_mask.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann.keypoints = keypoints[i]
                ann.num_keypoints = sum(k != 0 for k in keypoints[i][2::3])
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(
    dataset: Union[torch.utils.data.Subset, torchvision.datasets.CocoDetection]
) -> COCO:
    """

    :param dataset:
    :type dataset:
    :return:
    :rtype:"""
    for i in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    return convert_to_coco_api(dataset)


class CocoDetection(torchvision.datasets.CocoDetection):
    """ """

    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = NamedTensorTuple(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def get_coco_ins(
    root_path: Path,
    image_set: SplitEnum,
    transforms,
    mode: CocoModeEnum = CocoModeEnum.instances,
):
    """

    :param root_path:
    :type root_path:
    :param image_set:
    :type image_set:
    :param transforms:
    :type transforms:
    :param mode:
    :type mode:
    :return:
    :rtype:"""
    assert image_set in SplitEnum
    assert image_set != SplitEnum.testing

    annotations_path = Path("annotations")
    PATHS = {
        SplitEnum.training: (
            "train2017",
            annotations_path / f"{mode}_{'train'}2017.json",
        ),
        SplitEnum.validation: (
            "val2017",
            annotations_path / f"{mode}_{'val'}2017.json",
        ),
    }

    t = [ConvertCocoPolysToMask()]

    if transforms is not None:
        t.append(transforms)
    transforms = Compose(t)

    img_folder, ann_file = PATHS[image_set]

    dataset = CocoDetection(
        root_path / img_folder, root_path / ann_file, transforms=transforms
    )

    if image_set == SplitEnum.training:
        dataset = _coco_remove_images_without_annotations(dataset)

    # dataset = torch.utils.data.Subset(dataset, [i for i in range(500)])

    return dataset


def get_coco_kp(root, image_set, transforms):
    """

    :param root:
    :type root:
    :param image_set:
    :type image_set:
    :param transforms:
    :type transforms:
    :return:
    :rtype:"""
    return get_coco_ins(root, image_set, transforms, mode=CocoModeEnum.person_keypoints)
