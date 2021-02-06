#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 07/03/2020
           """

import numpy
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

__all__ = ["bb_from_mask", "get_model_instance_segmentation"]


def bb_from_mask(hard_mask):
    nz = numpy.nonzero(hard_mask)
    return [numpy.min(nz[0]), numpy.min(nz[1]), numpy.max(nz[0]), numpy.max(nz[1])]


def get_model_instance_segmentation(num_classes, hidden_layer: int = 256):
    """

    :param num_classes:
    :type num_classes:
    :return:
    :rtype:"""

    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model
