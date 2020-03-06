#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torchvision


__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 11/11/2019
           """

from draugr import (
    set_all_parameter_requires_grad,
    set_first_n_parameter_requires_grad,
    get_trainable_parameters,
)


def resnet_retrain(
    num_classes: int,
    freeze_first_num: int = 6,
    pretrained: bool = True,
    resnet_factory: callable = torchvision.models.resnet18,
):
    model = resnet_factory(pretrained=pretrained)
    if freeze_first_num == 0:
        set_all_parameter_requires_grad(model)
    elif freeze_first_num > 0:
        set_first_n_parameter_requires_grad(model, freeze_first_num)

    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    return model, get_trainable_parameters(model)
