#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/03/2020
           """

import copy
import json
import logging
from collections import defaultdict, namedtuple
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy
import pycocotools.mask
import torch
import torch._six
import torchvision
from draugr.python_utilities.exceptions import IncompatiblePackageVersions
from pycocotools.coco import COCO  # Version 2.0 REQUIRES numpy 1.17
from pycocotools.cocoeval import COCOeval

if pycocotools.coco.__version__ == "2.0" and "1.18" in numpy.__version__:
    print("Hint: downgrade numpy to 1.17.x")
    raise IncompatiblePackageVersions(
        numpy, "pycocotools", pycocotools=pycocotools.coco.__version__
    )

from draugr.torch_utilities import minmax_to_xywh_torch
from neodroidvision.utilities.torch_utilities.distributing.distributing_utilities import (
    all_gather_cuda,
)

__all__ = [
    "CocoEvaluator",
    "merge",
    "create_common_coco_eval",
    "create_index",
    "load_results",
    "coco_evaluation",
    "get_iou_types",
]

BboxPredTuple = namedtuple("BboxPredTuple", ("boxes", "scores", "labels"))
SegmPredTuple = namedtuple("SegmPredTuple", ("masks", "scores", "labels"))
KeypointsPredTuple = namedtuple("KeypointsPredTuple", ("keypoints", "scores", "labels"))


class IouType(Enum):
    BoundingBox = "bbox"
    Segmentation = "segm"
    Keypoints = "keypoints"


class CocoEvaluator(object):
    """ """

    def __init__(self, coco_api: COCO, iou_types: Sequence[IouType]):
        assert isinstance(iou_types, (list, tuple))
        self.coco_api = copy.deepcopy(coco_api)

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            assert iou_type in IouType
            self.coco_eval[iou_type] = COCOeval(self.coco_api, iouType=iou_type.value)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions: Dict) -> None:
        """

        :param predictions:
        :type predictions:"""
        img_ids = list(numpy.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare_data(predictions, iou_type)
            coco_dt = load_results(self.coco_api, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        """ """
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = numpy.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(
                self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type]
            )

    def accumulate(self):
        """ """
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        """ """
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()

    def prepare_data(
        self, predictions: Sequence, iou_type: IouType
    ) -> List[Dict[str, Any]]:
        """

        :param predictions:
        :type predictions:
        :param iou_type:
        :type iou_type:
        :return:
        :rtype:"""
        if iou_type == iou_type.BoundingBox:
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == iou_type.Segmentation:
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == iou_type.Keypoints:
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError(f"Unknown iou type {iou_type}")

    def prepare_for_coco_detection(self, predictions: Sequence[BboxPredTuple]):
        """

        :param predictions:
        :type predictions:
        :return:
        :rtype:"""
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = minmax_to_xywh_torch(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions: Sequence[SegmPredTuple]):
        """

        :param predictions:
        :type predictions:
        :return:
        :rtype:"""
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                pycocotools.mask.encode(
                    numpy.array(mask[0, :, :, numpy.newaxis], order="F")
                )[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions: Sequence[KeypointsPredTuple]):
        """

        :param predictions:
        :type predictions:
        :return:
        :rtype:"""
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def evaluate(iou_type_evaluator: COCOeval) -> Tuple:
    """
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None"""
    p = iou_type_evaluator.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = "segm" if p.useSegm == 1 else "bbox"
        print(f"useSegm (deprecated) is not None. Running {p.iouType} evaluation")
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(numpy.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(numpy.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    iou_type_evaluator.params = p

    iou_type_evaluator._prepare()
    # loop through images, area range, max detection number
    cat_ids = p.catIds if p.useCats else [-1]

    compute_iou = None
    if p.iouType == "segm" or p.iouType == "bbox":
        compute_iou = iou_type_evaluator.computeIoU
    elif p.iouType == "keypoints":
        compute_iou = iou_type_evaluator.computeOks

    iou_type_evaluator.ious = {
        (imgId, catId): compute_iou(imgId, catId)
        for imgId in p.imgIds
        for catId in cat_ids
    }

    evaluate_img = iou_type_evaluator.evaluateImg
    max_det = p.maxDets[-1]
    eval_imgs = [
        evaluate_img(img_id, cat_id, area_rng, max_det)
        for cat_id in cat_ids
        for area_rng in p.areaRng
        for img_id in p.imgIds
    ]

    eval_imgs = numpy.asarray(
        eval_imgs
    ).reshape(  # this is NOT in the pycocotools code, but could be done outside
        len(cat_ids), len(p.areaRng), len(p.imgIds)
    )
    iou_type_evaluator._paramsEval = copy.deepcopy(iou_type_evaluator.params)

    return p.imgIds, eval_imgs


def merge(img_ids, eval_imgs):
    """

    :param img_ids:
    :type img_ids:
    :param eval_imgs:
    :type eval_imgs:
    :return:
    :rtype:"""
    all_img_ids = all_gather_cuda(img_ids)
    all_eval_imgs = all_gather_cuda(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = numpy.array(merged_img_ids)
    merged_eval_imgs = numpy.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = numpy.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    """

    :param coco_eval:
    :type coco_eval:
    :param img_ids:
    :type img_ids:
    :param eval_imgs:
    :type eval_imgs:"""
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.img_ids = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################

# Ideally, pycocotools wouldn't have hard-coded prints
# so that we could avoid copy-pasting those two functions


def create_index(self):
    """

    :param self:
    :type self:"""
    # create index
    # print('creating index...')
    anns, cats, imgs = {}, {}, {}
    imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
    if "annotations" in self.dataset:
        for ann in self.dataset["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)
            anns[ann["id"]] = ann

    if "images" in self.dataset:
        for img in self.dataset["images"]:
            imgs[img["id"]] = img

    if "categories" in self.dataset:
        for cat in self.dataset["categories"]:
            cats[cat["id"]] = cat

    if "annotations" in self.dataset and "categories" in self.dataset:
        for ann in self.dataset["annotations"]:
            catToImgs[ann["category_id"]].append(ann["image_id"])

    # print('index created!')

    # create class members
    self.anns = anns
    self.imgToAnns = imgToAnns
    self.catToImgs = catToImgs
    self.imgs = imgs
    self.cats = cats


def load_results(self, resFile) -> COCO:
    """
    Load result file and return a result api object.
    :param   resFile (str)     : file name of result file
    :return: res (obj)         : result api object"""
    res = COCO()
    res.dataset["images"] = [img for img in self.dataset["images"]]

    # print('Loading and preparing results...')
    # tic = time.time()
    if isinstance(resFile, torch._six.string_classes):
        anns = json.load(open(resFile))
    elif type(resFile) == numpy.ndarray:
        anns = self.loadNumpyAnnotations(resFile)
    else:
        anns = resFile
    assert type(anns) == list, "results in not an array of objects"
    annsImgIds = [ann["image_id"] for ann in anns]
    assert set(annsImgIds) == (
        set(annsImgIds) & set(self.getImgIds())
    ), "Results do not correspond to current coco set"
    if "caption" in anns[0]:
        imgIds = set([img["id"] for img in res.dataset["images"]]) & set(
            [ann["image_id"] for ann in anns]
        )
        res.dataset["images"] = [
            img for img in res.dataset["images"] if img["id"] in imgIds
        ]
        for id, ann in enumerate(anns):
            ann["id"] = id + 1
    elif "bbox" in anns[0] and not anns[0]["bbox"] == []:
        res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
        for id, ann in enumerate(anns):
            bb = ann["bbox"]
            x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
            if "segmentation" not in ann:
                ann["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            ann["area"] = bb[2] * bb[3]
            ann["id"] = id + 1
            ann["iscrowd"] = 0
    elif "segmentation" in anns[0]:
        res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
        for id, ann in enumerate(anns):
            # now only support compressed RLE format as segmentation results
            ann["area"] = pycocotools.mask.area(ann["segmentation"])
            if "bbox" not in ann:
                ann["bbox"] = pycocotools.mask.toBbox(ann["segmentation"])
            ann["id"] = id + 1
            ann["iscrowd"] = 0
    elif "keypoints" in anns[0]:
        res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
        for id, ann in enumerate(anns):
            s = ann["keypoints"]
            x = s[0::3]
            y = s[1::3]
            x0, x1, y0, y1 = numpy.min(x), numpy.max(x), numpy.min(y), numpy.max(y)
            ann["area"] = (x1 - x0) * (y1 - y0)
            ann["id"] = id + 1
            ann["bbox"] = [x0, y0, x1 - x0, y1 - y0]
    # print('DONE (t={:0.2f}s)'.format(time.time()- tic))

    res.dataset["annotations"] = anns
    create_index(res)
    return res


#################################################################
# end of straight copy from pycocotools, just removing the prints
#################################################################
def coco_evaluation(dataset, predictions, output_dir: Path, iteration=None):
    """

    :param dataset:
    :type dataset:
    :param predictions:
    :type predictions:
    :param output_dir:
    :type output_dir:
    :param iteration:
    :type iteration:
    :return:
    :rtype:"""
    coco_results = []
    for i, prediction in enumerate(predictions):
        img_info = dataset.get_img_info(i)
        prediction = prediction.resize((img_info["width"], img_info["height"])).numpy()
        boxes, labels, scores = (
            prediction["boxes"],
            prediction["labels"],
            prediction["scores"],
        )

        image_id, annotation = dataset.get_annotation(i)
        class_mapper = dataset.contiguous_id_to_coco_id
        if labels.shape[0] == 0:
            continue

        boxes = boxes.tolist()
        labels = labels.tolist()
        scores = scores.tolist()
        coco_results.extend(
            [
                {
                    "image_id": image_id,
                    "category_id": class_mapper[labels[k]],
                    "bbox": [
                        box[0],
                        box[1],
                        box[2] - box[0],
                        box[3] - box[1],
                    ],  # to xywh format
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    iou_type = "bbox"
    json_result_file = str(output_dir / f"{iou_type}.json")
    logger = logging.getLogger("SSD.inference")
    logger.info(f"Writing results to {json_result_file}...")
    with open(json_result_file, "w") as f:
        json.dump(coco_results, f)

    coco_gt = dataset.coco
    coco_dt = coco_gt.load_results(json_result_file)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    result_strings = []
    keys = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
    metrics = {}
    for i, key in enumerate(keys):
        metrics[key] = coco_eval.stats[i]
        logger.info(f"{key:<10}: {round(coco_eval.stats[i], 3)}")
        result_strings.append(f"{key:<10}: {round(coco_eval.stats[i], 3)}")

    if iteration is not None:
        result_path = str(output_dir / f"result_{iteration:07d}.txt")
    else:
        result_path = str(
            output_dir / f"result_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        )
    with open(result_path, "w") as f:
        f.write("\n".join(result_strings))

    return dict(metrics=metrics)


def get_iou_types(model) -> Sequence[IouType]:
    """

    :param model:
    :type model:
    :return:
    :rtype:"""
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module

    iou_types = [IouType.BoundingBox]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append(IouType.Segmentation)
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append(IouType.Keypoints)
    return iou_types
