#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from draugr.torch_utilities.to_tensor import to_tensor

__author__ = 'Christian Heider Nielsen'
__doc__ = r'''

           Created on 08/10/2019
           '''


def to_device_tensor_iterator(data_iterator, device):
  while True:
    yield (to_tensor(i, device=device) for i in next(data_iterator))


if __name__ == '__main__':
  import numpy

  a = iter(numpy.random.sample((5, 5, 5)))
  for a in to_device_tensor_iterator(a, 'cpu'):
    d, *_ = a
    print(d)
    print(type(d))
