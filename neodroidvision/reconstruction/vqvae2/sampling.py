#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random

__author__ = 'cnheider'
__doc__ = r'''
           '''


def random_sample_softmax(probs):
  number = random.random()
  for i, x in enumerate(probs):
    number -= x
    if number <= 0:
      return i
  return len(probs) - 1
