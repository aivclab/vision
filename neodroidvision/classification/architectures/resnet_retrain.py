#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List, Tuple

import torch
import torchvision

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 11/11/2019
           """

from draugr.torch_utilities import (
    set_all_parameter_requires_grad,
    set_first_n_parameter_requires_grad,
    trainable_parameters,
)

__all__ = ["resnet_retrain"]

from torch.nn.parameter import Parameter

from torchvision.models import ResNet


def resnet_retrain(
    num_classes: int,
    freeze_first_num: int = 6,
    pretrained: bool = True,
    resnet_factory: callable = torchvision.models.resnet18,
) -> Tuple[ResNet, List[Parameter]]:
    """

    Args:
      num_classes:
      freeze_first_num:
      pretrained:
      resnet_factory:

    Returns:

    """
    model = resnet_factory(pretrained=pretrained)
    if freeze_first_num == 0:
        set_all_parameter_requires_grad(model)
    elif freeze_first_num > 0:
        set_first_n_parameter_requires_grad(model, freeze_first_num)

    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    return model, trainable_parameters(model)
