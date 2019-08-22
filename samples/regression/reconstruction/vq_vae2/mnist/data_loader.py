#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'cnheider'
__doc__ = r'''
           '''

import torch
import torchvision.datasets
import torchvision.transforms

from neodroidvision import PROJECT_APP_PATH


def load_images(train=True):
  while True:
    for data, _ in create_data_loader(train):
      yield data


def create_data_loader(train, BATCH_SIZE=32):
  mnist = torchvision.datasets.MNIST(PROJECT_APP_PATH.user_cache / 'data',
                                     train=train, download=True,
                                     transform=torchvision.transforms.ToTensor())
  return torch.utils.data.DataLoader(mnist,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True)
