#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torchvision

from neodroidvision.classification import set_all_parameter_requires_grad
from neodroidvision.classification.architectures.retrain_utilities import set_first_n_parameter_requires_grad

__author__ = 'Christian Heider Nielsen'
__doc__ = r'''

           Created on 11/11/2019
           '''


def resnet_retrain(num_classes,
                   freeze=6,
                   pretrained=True,
                   resnet_version=torchvision.models.resnet18):
  model = resnet_version(pretrained=pretrained)
  if freeze == 0:
    set_all_parameter_requires_grad(model)
  elif freeze > 0:
    set_first_n_parameter_requires_grad(model, freeze)

  model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

  params_to_update = []
  for name, param in model.named_parameters():
    if param.requires_grad == True:
      params_to_update.append(param)

  return model, params_to_update
