#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 23/03/2020
           """

import random

from torchvision.transforms.functional import to_tensor

__all__ = ["TupleToTensor", "TupleCompose", "TupleRandomHorizontalFlip"]


class TupleCompose(object):
    """ """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class TupleRandomHorizontalFlip(object):
    """ """

    @staticmethod
    def _flip_coco_person_keypoints(kps, width):
        flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
        flipped_data = kps[:, flip_inds]
        flipped_data[..., 0] = width - flipped_data[..., 0]
        # Maintain COCO convention that if visibility == 0, then x, y = 0
        inds = flipped_data[..., 2] == 0
        flipped_data[inds] = 0
        return flipped_data

    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = self._flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class TupleToTensor(object):
    def __call__(self, image, target):
        return to_tensor(image), target