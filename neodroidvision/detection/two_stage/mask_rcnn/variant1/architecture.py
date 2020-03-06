#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 28/02/2020
           """

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_pretrained_instance_segmentation_model(num_classes, hidden_layer=256):
    """
  load an instance segmentation model pre-trained on COCO

  :param num_classes:
  :param hidden_layer:
  :return:
  """

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, num_classes
    )

    # now get the number of input features for the mask classifier and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        model.roi_heads.mask_predictor.conv5_mask.in_channels, hidden_layer, num_classes
    )

    return model
