#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from multiprocessing import Pipe, Process

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import time
from neodroid.wrappers.observation_wrapper.observation_wrapper import CameraObservationWrapper
from segmentation.segmentation_utilities.plot_utilities import channel_transform

__author__ = 'cnheider'

import torch

a_transform = transforms.Compose([
  transforms.Resize(224),
  transforms.CenterCrop(224),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor()
  ])

a_retransform = transforms.Compose([
  transforms.ToPILImage('RGB')
  ])


def NeodroidClassificationGenerator(env, device, batch_size=64):
  while True:
    predictors = []
    class_responses = []
    while len(predictors) < batch_size:
      state = env.update()
      rgb_arr = state.observer('RGB').observation_value
      rgb_arr = Image.open(rgb_arr).convert('RGB')
      a_class = state.observer('Class').observation_value

      predictors.append(a_transform(rgb_arr))
      class_responses.append(int(a_class))

    a = torch.stack(predictors).to(device)
    b = torch.LongTensor(class_responses).to(device)
    yield a, b


def VestasClassificationGenerator(
    batch_size=6,
    workers=1,
    path='/home/heider/Data/Datasets/Vision/vestas'):
  train_dataset = datasets.ImageFolder(path, a_transform)

  if False:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
  else:
    train_sampler = None

  torch.manual_seed(time.time())
  train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             shuffle=(train_sampler is None),
                                             num_workers=workers,
                                             pin_memory=True,
                                             sampler=train_sampler)

  return train_loader





if __name__ == '__main__':
  a = VestasClassificationGenerator()
  for i, (g, c) in enumerate(a):
    print(c)

  '''

  neodroid_generator = NeodroidDataGenerator()
  train_loader = torch.utils.data.DataLoader(dataset=neodroid_generator,
                                             batch_size=12,
                                             shuffle=True)
  for p, r in train_loader:
    print(r)
  '''
