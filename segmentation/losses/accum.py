#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn.functional as F
from warg import NOD

from segmentation.losses.dice_loss import dice_loss
from segmentation.losses.jaccard_loss import jaccard_loss

__author__ = 'cnheider'

import torch
import warg


def calculate_loss(pred,
                   target,
                   reconstruction,
                   original):

  seg_bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target)
  ae_bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(reconstruction, original)

  pred_soft = torch.sigmoid(pred)
  dice = dice_loss(pred_soft, target, epsilon=1)
  jaccard = jaccard_loss(pred_soft, target, epsilon=1)

  terms = NOD.dict_of(dice, jaccard, ae_bce_loss, seg_bce_loss)

  term_weight = 1 / len(terms)
  weighted_terms = [term.mean() * term_weight for term in terms.as_list()]

  loss = sum(weighted_terms)

  return NOD.dict_of(loss, terms)
