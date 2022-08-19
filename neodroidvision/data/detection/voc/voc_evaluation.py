#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/03/2020
           """

import itertools
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy
import six

__all__ = [
    "bbox_iou",
    "eval_detection_voc",
    "calc_detection_voc_ap",
    "calc_detection_voc_prec_rec",
    "voc_evaluation",
]


def bbox_iou(bbox_a: numpy.ndarray, bbox_b: numpy.ndarray) -> numpy.ndarray:
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.

IoU is calculated as a ratio of area of the intersection
and area of the union.
This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
same type.
The output is same type as the type of the inputs.
Args:
bbox_a (array): An array whose shape is :math:`(N, 4)`.
:math:`N` is the number of bounding boxes.
The dtype should be :obj:`numpy.float32`.
bbox_b (array): An array similar to :obj:`bbox_a`,
whose shape is :math:`(K, 4)`.
The dtype should be :obj:`numpy.float32`.
Returns:
array:
An array whose shape is :math:`(N, K)`. \
An element at index :math:`(n, k)` contains IoUs between \
:math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
box in :obj:`bbox_b`.
"""
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    tl = numpy.maximum(bbox_a[:, None, :2], bbox_b[:, :2])  # top left

    br = numpy.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])  # bottom right

    area_i = numpy.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = numpy.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = numpy.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def eval_detection_voc(
    pred_bboxes,
    pred_labels,
    pred_scores,
    gt_bboxes,
    gt_labels,
    gt_difficults=None,
    iou_thresh: float = 0.5,
    use_07_metric=False,
) -> Tuple:
    """Calculate average precisions based on evaluation code of PASCAL VOC.

This function evaluates predicted bounding boxes obtained from a dataset
which has :math:`N` images by using average precision for each class.
The code is based on the evaluation code used in PASCAL VOC Challenge.

Args:
pred_bboxes (iterable of numpy.ndarray): An iterable of :math:`N`
sets of bounding boxes.
Its index corresponds to an index for the base dataset.
Each element of :obj:`pred_bboxes` is a set of coordinates
of bounding boxes. This is an array whose shape is :math:`(R, 4)`,
where :math:`R` corresponds
to the number of bounding boxes, which may vary among boxes.
The second axis corresponds to
:math:`y_{min}, x_{min}, y_{max}, x_{max}` of a bounding box.
pred_labels (iterable of numpy.ndarray): An iterable of labels.
Similar to :obj:`pred_bboxes`, its index corresponds to an
index for the base dataset. Its length is :math:`N`.
pred_scores (iterable of numpy.ndarray): An iterable of confidence
scores for predicted bounding boxes. Similar to :obj:`pred_bboxes`,
its index corresponds to an index for the base dataset.
Its length is :math:`N`.
gt_bboxes (iterable of numpy.ndarray): An iterable of ground truth
bounding boxes
whose length is :math:`N`. An element of :obj:`gt_bboxes` is a
bounding box whose shape is :math:`(R, 4)`. Note that the number of
bounding boxes in each image does not need to be same as the number
of corresponding predicted boxes.
gt_labels (iterable of numpy.ndarray): An iterable of ground truth
labels which are organized similarly to :obj:`gt_bboxes`.
gt_difficults (iterable of numpy.ndarray): An iterable of boolean
arrays which is organized similarly to :obj:`gt_bboxes`.
This tells whether the
corresponding ground truth bounding box is difficult or not.
By default, this is :obj:`None`. In that case, this function
considers all bounding boxes to be not difficult.
iou_thresh (float): A prediction is correct if its Intersection over
Union with the ground truth is above this value.
use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
for calculating average precision. The default value is
:obj:`False`.

Returns:
dict:

The keys, value-types and the description of the values are listed
below.

* **ap** (*numpy.ndarray*): An array of average precisions. \
The :math:`l`-th value corresponds to the average precision \
for class :math:`l`. If class :math:`l` does not exist in \
either :obj:`pred_labels` or :obj:`gt_labels`, the corresponding \
value is set to :obj:`numpy.nan`.
* **map** (*float*): The mean of Average Precisions over classes.

"""

    prec, rec = calc_detection_voc_prec_rec(
        pred_bboxes,
        pred_labels,
        pred_scores,
        gt_bboxes,
        gt_labels,
        gt_difficults,
        iou_thresh=iou_thresh,
    )

    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)

    return ap, numpy.nanmean(ap)  # Mean Average Precision


def calc_detection_voc_prec_rec(
    pred_bboxes,
    pred_labels,
    pred_scores,
    gt_bboxes,
    gt_labels,
    gt_difficults=None,
    iou_thresh: float = 0.5,
) -> Tuple:
    """Calculate precision and recall based on evaluation code of PASCAL VOC.

This function calculates precision and recall of
predicted bounding boxes obtained from a dataset which has :math:`N`
images.
The code is based on the evaluation code used in PASCAL VOC Challenge.

Args:
pred_bboxes (iterable of numpy.ndarray): An iterable of :math:`N`
sets of bounding boxes.
Its index corresponds to an index for the base dataset.
Each element of :obj:`pred_bboxes` is a set of coordinates
of bounding boxes. This is an array whose shape is :math:`(R, 4)`,
where :math:`R` corresponds
to the number of bounding boxes, which may vary among boxes.
The second axis corresponds to
:math:`y_{min}, x_{min}, y_{max}, x_{max}` of a bounding box.
pred_labels (iterable of numpy.ndarray): An iterable of labels.
Similar to :obj:`pred_bboxes`, its index corresponds to an
index for the base dataset. Its length is :math:`N`.
pred_scores (iterable of numpy.ndarray): An iterable of confidence
scores for predicted bounding boxes. Similar to :obj:`pred_bboxes`,
its index corresponds to an index for the base dataset.
Its length is :math:`N`.
gt_bboxes (iterable of numpy.ndarray): An iterable of ground truth
bounding boxes
whose length is :math:`N`. An element of :obj:`gt_bboxes` is a
bounding box whose shape is :math:`(R, 4)`. Note that the number of
bounding boxes in each image does not need to be same as the number
of corresponding predicted boxes.
gt_labels (iterable of numpy.ndarray): An iterable of ground truth
labels which are organized similarly to :obj:`gt_bboxes`.
gt_difficults (iterable of numpy.ndarray): An iterable of boolean
arrays which is organized similarly to :obj:`gt_bboxes`.
This tells whether the
corresponding ground truth bounding box is difficult or not.
By default, this is :obj:`None`. In that case, this function
considers all bounding boxes to be not difficult.
iou_thresh (float): A prediction is correct if its Intersection over
Union with the ground truth is above this value..

Returns:
tuple of two lists:
This function returns two lists: :obj:`prec` and :obj:`rec`.

* :obj:`prec`: A list of arrays. :obj:`prec[l]` is precision \
for class :math:`l`. If class :math:`l` does not exist in \
either :obj:`pred_labels` or :obj:`gt_labels`, :obj:`prec[l]` is \
set to :obj:`None`.
* :obj:`rec`: A list of arrays. :obj:`rec[l]` is recall \
for class :math:`l`. If class :math:`l` that is not marked as \
difficult does not exist in \
:obj:`gt_labels`, :obj:`rec[l]` is \
set to :obj:`None`.

"""

    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes = iter(gt_bboxes)
    gt_labels = iter(gt_labels)
    if gt_difficults is None:
        gt_difficults = itertools.repeat(None)
    else:
        gt_difficults = iter(gt_difficults)

    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)

    for (
        pred_bbox,
        pred_label,
        pred_score,
        gt_bbox,
        gt_label,
        gt_difficult,
    ) in six.moves.zip(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults
    ):

        if gt_difficult is None:
            gt_difficult = numpy.zeros(gt_bbox.shape[0], dtype=bool)

        for l in numpy.unique(numpy.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += numpy.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1

            iou = bbox_iou(pred_bbox_l, gt_bbox_l)
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = numpy.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    for iter_ in (
        pred_bboxes,
        pred_labels,
        pred_scores,
        gt_bboxes,
        gt_labels,
        gt_difficults,
    ):
        if next(iter_, None) is not None:
            raise ValueError("Length of input iterables need to be same.")

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = numpy.array(score[l])
        match_l = numpy.array(match[l], dtype=numpy.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = numpy.cumsum(match_l == 1)
        fp = numpy.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec


def calc_detection_voc_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.

    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.

    Args:
    prec (list of numpy.array): A list of arrays.
    :obj:`prec[l]` indicates precision for class :math:`l`.
    If :obj:`prec[l]` is :obj:`None`, this function returns
    :obj:`numpy.nan` for class :math:`l`.
    rec (list of numpy.array): A list of arrays.
    :obj:`rec[l]` indicates recall for class :math:`l`.
    If :obj:`rec[l]` is :obj:`None`, this function returns
    :obj:`numpy.nan` for class :math:`l`.
    use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
    for calculating average precision. The default value is
    :obj:`False`.

    Returns:
    ~numpy.ndarray:
    This function returns an array of average precisions.
    The :math:`l`-th value corresponds to the average precision
    for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
    :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
    """

    n_fg_class = len(prec)
    ap = numpy.empty(n_fg_class)
    for l in six.moves.range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = numpy.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in numpy.arange(0.0, 1.1, 0.1):
                if numpy.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = numpy.max(numpy.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = numpy.concatenate(([0], numpy.nan_to_num(prec[l]), [0]))
            mrec = numpy.concatenate(([0], rec[l], [1]))

            mpre = numpy.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = numpy.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = numpy.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def voc_evaluation(dataset, predictions, output_dir: Path, iteration=None):
    """

    :param dataset:
    :param predictions:
    :param output_dir:
    :param iteration:
    :return:
    """
    class_names = dataset.class_names

    pred_boxes_list = []
    pred_labels_list = []
    pred_scores_list = []
    gt_boxes_list = []
    gt_labels_list = []
    gt_difficults = []

    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, gt_labels, is_difficult = annotation
        gt_boxes_list.append(gt_boxes)
        gt_labels_list.append(gt_labels)
        gt_difficults.append(is_difficult.astype(numpy.bool))

        img_info = dataset.get_img_info(i)
        prediction = predictions[i]
        prediction = prediction.resize((img_info["width"], img_info["height"])).numpy()
        boxes, labels, scores = (
            prediction["boxes"],
            prediction["labels"],
            prediction["scores"],
        )

        pred_boxes_list.append(boxes)
        pred_labels_list.append(labels)
        pred_scores_list.append(scores)
    ap, map = eval_detection_voc(
        pred_bboxes=pred_boxes_list,
        pred_labels=pred_labels_list,
        pred_scores=pred_scores_list,
        gt_bboxes=gt_boxes_list,
        gt_labels=gt_labels_list,
        gt_difficults=gt_difficults,
        iou_thresh=0.5,
        use_07_metric=True,
    )
    logger = logging.getLogger("SSD.inference")
    result_str = f"mAP: {map:.4f}\n"
    metrics = {"mAP": map}
    for i, ap in enumerate(ap):
        if i == 0:  # skip background
            continue
        metrics[class_names[i]] = ap
        result_str += f"{class_names[i]:<16}: {ap:.4f}\n"
    logger.info(result_str)

    if iteration is not None:
        result_path = str(output_dir / f"result_{iteration:07d}.txt")
    else:
        result_path = str(
            output_dir / f'result_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt'
        )
    with open(result_path, "w") as f:
        f.write(result_str)

    return dict(metrics=metrics)
