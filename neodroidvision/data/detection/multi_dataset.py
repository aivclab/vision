#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/03/2020
           """

from abc import abstractmethod
from pathlib import Path
from typing import Sequence, Tuple

from draugr.numpy_utilities import SplitEnum
from torch.utils.data import ConcatDataset

__all__ = ["MultiDataset"]

from draugr.torch_utilities import SupervisedDataset
from neodroidvision.detection.single_stage.ssd.bounding_boxes.ssd_transforms import (
    SSDTransform,
    SSDAnnotationTransform,
)


class MultiDataset(SupervisedDataset):
    """ """

    @property
    @abstractmethod
    def categories(self) -> Sequence:
        """ """
        raise NotImplementedError

    @property
    def response_shape(self) -> Tuple[int, ...]:
        """

        :return:
        :rtype:"""
        return (len(self.categories),)

    def __init__(
        self,
        *,
        cfg,
        dataset_type: callable,
        data_root: Path,
        sub_datasets: Tuple,
        split: SplitEnum = SplitEnum.training
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
        :rtype:"""
        super().__init__()
        assert len(sub_datasets) > 0, "No data found!"

        img_transform = SSDTransform(
            image_size=cfg.input.image_size,
            pixel_mean=cfg.input.pixel_mean,
            split=split,
        )

        if split == SplitEnum.training:
            annotation_transform = SSDAnnotationTransform(
                image_size=cfg.input.image_size,
                priors_cfg=cfg.model.box_head.priors,
                center_variance=cfg.model.box_head.center_variance,
                size_variance=cfg.model.box_head.size_variance,
                iou_threshold=cfg.model.box_head.iou_threshold,
            )
        else:
            annotation_transform = None

        datasets = []

        for dataset_name in sub_datasets:
            datasets.append(
                dataset_type(
                    data_root=data_root,
                    dataset_name=dataset_name,
                    split=split,
                    img_transform=img_transform,
                    annotation_transform=annotation_transform,
                )
            )

        # for testing, return a list of datasets
        if not split == SplitEnum.training:
            self.sub_datasets = datasets
        else:
            dataset = datasets[0]
            if len(datasets) > 1:
                dataset = ConcatDataset(datasets)

            self.sub_datasets = [dataset]
