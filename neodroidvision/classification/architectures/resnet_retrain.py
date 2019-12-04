#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torchvision

from neodroidvision.classification import set_all_parameter_requires_grad
from neodroidvision.classification.architectures.retrain_utilities import (
  set_first_n_parameter_requires_grad,
  )

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 11/11/2019
           """


def resnet_retrain(
  num_classes: int,
  freeze_first_num: int = 6,
  pretrained: bool = True,
  resnet_factory: callable = torchvision.models.resnet18
  ):
  model = resnet_factory(pretrained=pretrained)
  if freeze_first_num == 0:
    set_all_parameter_requires_grad(model)
  elif freeze_first_num > 0:
    set_first_n_parameter_requires_grad(model, freeze_first_num)

  model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

  params_to_update = []
  for name, param in model.named_parameters():
    if param.requires_grad == True:
      params_to_update.append(param)

  return model, params_to_update
