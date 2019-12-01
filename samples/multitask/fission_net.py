#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Christian Heider Nielsen'
__doc__ = r'''

           Created on 19/10/2019
           '''


def no_gradient_fusion():
  ''' calculate gradient per individual loss function and optimisers and pass on gradient updates
  seperately to the encoder'''
  pass


def late_gradient_fusion():
  ''' calculate gradient per individual loss function and optimisers and but collect and combine gradients at
  encoder'''
  pass


def early_gradient_fusion():
  ''' calculate gradient as collective loss function and pass gradients on through encoder with one monolithic
  optimiser'''
  pass
