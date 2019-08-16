#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch.nn.functional import mse_loss

__author__ = 'cnheider'
__doc__ = r'''
Objective functions for beta vae's
           '''

def reconstruction_loss(reconstruction, original):
  batch_size = original.size(0)
  assert batch_size != 0

  # recon_loss = F.binary_cross_entropy_with_logits(reconstruction,
  # original,
  # size_average=False).div(batch_size)

  # reconstruction = torch.sigmoid(reconstruction)
  recon_loss = mse_loss(reconstruction, original, size_average=False).div(batch_size)

  return recon_loss


def kl_divergence(mean, log_var):
  batch_size = mean.size(0)
  assert batch_size != 0
  if mean.data.ndimension() == 4:
    mean = mean.view(mean.size(0), mean.size(1))
  if log_var.data.ndimension() == 4:
    log_var = log_var.view(log_var.size(0), log_var.size(1))

  klds = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
  total_kld = klds.sum(1).mean(0, True)
  # dimension_wise_kld = klds.mean(0)
  # mean_kld = klds.mean(1).mean(0, True)

  return total_kld  # , dimension_wise_kld, mean_kld