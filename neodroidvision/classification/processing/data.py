#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from draugr.torch_utilities.initialisation.seeding import global_torch_device
from warg.pooled_queue_processor import PooledQueueProcessor, PooledQueueTask

__author__ = 'Christian Heider Nielsen'

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
      rgb_arr = state.sensor('RGB').value
      rgb_arr = Image.open(rgb_arr).convert('RGB')
      a_class = state.sensor('Class').value

      predictors.append(a_transform(rgb_arr))
      class_responses.append(int(a_class))

    a = torch.stack(predictors).to(global_torch_device())
    b = torch.LongTensor(class_responses).to(global_torch_device())
    yield a, b


class FetchConvert(PooledQueueTask):
  def __init__(self, env, device='cpu', batch_size=64, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.env = env
    self.batch_size = batch_size
    self.device = device

  def call(self, *args, **kwargs):
    predictors = []
    class_responses = []

    while len(predictors) < self.batch_size:
      state = self.env.update()
      rgb_arr = state.sensor('RGB').value
      rgb_arr = Image.open(rgb_arr).convert('RGB')
      a_class = state.sensor('Class').value

      predictors.append(a_transform(rgb_arr))
      class_responses.append(int(a_class))

    a = torch.stack(predictors).to(self.device)
    b = torch.LongTensor(class_responses).to(self.device)
    return (a, b)


def NeodroidClassificationGenerator2(env, device, batch_size=64):
  task = FetchConvert(env, device=device, batch_size=batch_size)

  processor = PooledQueueProcessor(task,
                                   fill_at_construction=True,
                                   max_queue_size=16,
                                   n_proc=None)

  for a in zip(processor):
    yield a


if __name__ == '__main__':
  neodroid_generator = NeodroidDataGenerator()
  train_loader = torch.utils.data.DataLoader(dataset=neodroid_generator,
                                             batch_size=12,
                                             shuffle=True)
  for p, r in train_loader:
    print(r)
