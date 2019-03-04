#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch.nn.functional as F
from warg import NOD
import neodroid as neo
from segmentation.segmentation_utilities.plot_utilities import channel_transform

__author__ = 'cnheider'

import torch


def neodroid_batch_data_iterator(env, device, batch_size=12):
  while True:
    predictors = []
    class_responses = []
    while len(predictors) < batch_size:
      info = env.update()
      rgb_arr = env.sensor('RGBCameraObserver')

      a_class = info.observer('InnerRotatorCategoricalObserver').observation_value

      predictors.append(channel_transform(rgb_arr))
      class_responses.append(to_one_hot(4,a_class))
    yield torch.FloatTensor(predictors).to(device), torch.FloatTensor(class_responses).to(device)


def calculate_loss(pred, true):

  loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, true)

  return NOD.dict_of(loss)
