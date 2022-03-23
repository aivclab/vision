#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 05/03/2020
           """

from neodroidvision.utilities import load_state_dict_from_url
from .efficient_net import EfficientNet
from .mobilenet import MobileNetV2
from .vgg import VGG

__all__ = ["efficient_net_b3_factory", "vgg_factory", "mobilenet_v2_factory"]


def efficient_net_b3_factory(image_size, pretrained=True):
    """

    Args:
      image_size:
      pretrained:

    Returns:

    """
    if pretrained:
        model = EfficientNet.from_pretrained("efficientnet-b3")
    else:
        model = EfficientNet.from_name("efficientnet-b3")
    return model


def vgg_factory(IMAGE_SIZE, pretrained=True):
    """

    Args:
      IMAGE_SIZE:
      pretrained:

    Returns:

    """
    model = VGG(IMAGE_SIZE)
    if pretrained:
        model.init_from_pretrain(
            load_state_dict_from_url(
                "https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth"
            )
        )
    return model


def mobilenet_v2_factory(image_size, pretrained=True):
    """

    Args:
      image_size:
      pretrained:

    Returns:

    """
    model = MobileNetV2(image_size)
    if pretrained:
        model.load_state_dict(
            load_state_dict_from_url(
                "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth"
            ),
            strict=False,
        )
    return model
