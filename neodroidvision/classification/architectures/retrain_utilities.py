#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Christian Heider Nielsen'
__doc__ = r'''

           Created on 13/11/2019
           '''


def set_all_parameter_requires_grad(model, bo=False):
  for param in model.parameters():
    param.requires_grad = bo


def set_first_n_parameter_requires_grad(model, n=6, bo=False):
  for i, child in enumerate(model.children()):
    if i <= n:
      set_all_parameter_requires_grad(child, bo)
