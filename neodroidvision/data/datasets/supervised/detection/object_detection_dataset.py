#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/03/2020
           """

from pathlib import Path
from typing import Tuple

from torch.utils.data import ConcatDataset

from neodroidvision.data.datasets.supervised.splitting import Split

__all__ = ["ObjectDetectionDataset"]

from neodroidvision.detection.single_stage.ssd.bounding_boxes.ssd_transforms import (
    SSDTransform,
    SSDTargetTransform,
)


class ObjectDetectionDataset:  # (SupervisedDataset):
    def __init__(
        self,
        *,
        cfg,
        dataset_type: callable,
        data_root: Path,
        sub_datasets: Tuple,
        split: Split = Split.Training
    ):
        """

    :param data_root:
    :type data_root:
    :param sub_datasets:
    :type sub_datasets:
    :param transform:
    :type transform:
    :param target_transform:
    :type target_transform:
    :param split:
    :type split:
    :return:
    :rtype:
    """
        assert len(sub_datasets) > 0, "No data found!"

        transform = (
            SSDTransform(
                IMAGE_SIZE=cfg.INPUT.IMAGE_SIZE,
                PIXEL_MEAN=cfg.INPUT.PIXEL_MEAN,
                split=split,
            ),
        )
        target_transform = (
            SSDTargetTransform(
                cfg.INPUT.IMAGE_SIZE,
                cfg.MODEL.PRIORS,
                cfg.MODEL.CENTER_VARIANCE,
                cfg.MODEL.SIZE_VARIANCE,
                cfg.MODEL.THRESHOLD,
            )
            if split == Split.Training
            else None
        )

        datasets = []

        for dataset_name in sub_datasets:
            datasets.append(
                dataset_type(
                    data_root=data_root,
                    dataset_name=dataset_name,
                    split=split,
                    transform=transform,
                    target_transform=target_transform,
                )
            )

        # for testing, return a list of datasets
        if not split == Split.Training:
            self.sub_datasets = datasets
        else:
            dataset = datasets[0]
            if len(datasets) > 1:
                dataset = ConcatDataset(datasets)

            self.sub_datasets = [dataset]
