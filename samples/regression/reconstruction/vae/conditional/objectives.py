#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

__author__ = 'cnheider'
__doc__ = r'''
           '''


def loss_fn(recon_x,
            x,
            mean,
            log_var):
  BCE = torch.nn.functional.binary_cross_entropy(recon_x.view(-1, 28 * 28),
                                                 x.view(-1, 28 * 28),
                                                 reduction='sum')
  KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

  return (BCE + KLD) / x.size(0)
