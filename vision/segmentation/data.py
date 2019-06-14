#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from warg.named_ordered_dictionary import NOD

from vision.segmentation.loss_functions.dice_loss import dice_loss
from vision.segmentation.loss_functions.jaccard_loss import jaccard_loss
from vision.segmentation.segmentation_utilities.plot_utilities import channel_transform

__author__ = 'cnheider'


def neodroid_batch_data_iterator(env, device, batch_size=12):
  while True:
    predictors = []
    mask_responses = []
    depth_responses = []
    normals_responses = []
    while len(predictors) < batch_size:
      env.update_models()
      rgb_arr = env.sensor('RGBCameraObserver')
      seg_arr = env.sensor('LayerSegmentationCameraObserver')
      depth_arr = env.sensor('CompressedDepthCameraObserver')
      mul_depth_arr = env.sensor('DepthCameraObserver')
      normal_arr = env.sensor('NormalCameraObserver')

      red_mask = np.zeros(seg_arr.shape[:-1])
      green_mask = np.zeros(seg_arr.shape[:-1])
      blue_mask = np.zeros(seg_arr.shape[:-1])

      reddish = seg_arr[:, :, 0] > 50
      greenish = seg_arr[:, :, 1] > 50
      blueish = seg_arr[:, :, 2] > 50

      red_mask[reddish] = 1
      green_mask[greenish] = 1
      blue_mask[blueish] = 1

      depth_image = np.zeros(depth_arr.shape[:-1])

      depth_image[:, :] = depth_arr[..., 1]

      predictors.append(channel_transform(rgb_arr))
      mask_responses.append(np.asarray([red_mask, blue_mask, green_mask]))
      depth_responses.append(np.clip(np.asarray([depth_image / 255.0]), 0, 1))
      normals_responses.append(channel_transform(normal_arr))
    yield torch.FloatTensor(predictors).to(device), (torch.FloatTensor(mask_responses).to(device),
                                                     torch.FloatTensor(depth_responses).to(device),
                                                     torch.FloatTensor(normals_responses).to(device))


def calculate_loss(seg, recon, depth, normals):
  ((seg_pred, seg_target),
   (recon_pred, recon_target),
   (depth_pred, depth_target),
   (normals_pred, normals_target)) = (seg, recon, depth, normals)

  seg_bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(seg_pred, seg_target)
  ae_bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(recon_pred, recon_target)
  normals_bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(normals_pred, normals_target)
  depth_bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(depth_pred, depth_target)

  pred_soft = torch.sigmoid(seg_pred)
  dice = dice_loss(pred_soft, seg_target, epsilon=1)
  jaccard = jaccard_loss(pred_soft, seg_target, epsilon=1)

  terms = NOD.nod_of(dice,
                     jaccard,
                     ae_bce_loss,
                     seg_bce_loss,
                     depth_bce_loss,
                     normals_bce_loss
                     )

  term_weight = 1 / len(terms)
  weighted_terms = [term.mean() * term_weight for term in terms.as_list()]

  loss = sum(weighted_terms)

  return NOD.nod_of(loss, terms)
