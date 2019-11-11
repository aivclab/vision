#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import torch
import torch.utils.data

from draugr.torch_utilities import channel_transform
from draugr.torch_utilities.to_tensor import to_tensor
from neodroidvision.segmentation import dice_loss, jaccard_loss
from warg.named_ordered_dictionary import NOD

__author__ = 'Christian Heider Nielsen'


def neodroid_camera_data_iterator(env,
                                  device,
                                  batch_size=12):
  while True:
    rgb = []
    mask_responses = []
    depth_responses = []
    normals_responses = []
    while len(rgb) < batch_size:
      env.update()
      rgb_arr = env.sensor('RGB')
      seg_arr = env.sensor('Layer')
      depth_arr = env.sensor('CompressedDepth')
      normal_arr = env.sensor('Normal')

      red_mask = numpy.zeros(seg_arr.shape[:-1])
      green_mask = numpy.zeros(seg_arr.shape[:-1])
      blue_mask = numpy.zeros(seg_arr.shape[:-1])
      # alpha_mask = numpy.ones(seg_arr.shape[:-1])

      reddish = seg_arr[:, :, 0] > 50
      greenish = seg_arr[:, :, 1] > 50
      blueish = seg_arr[:, :, 2] > 50

      red_mask[reddish] = 1
      green_mask[greenish] = 1
      blue_mask[blueish] = 1

      depth_image = numpy.zeros(depth_arr.shape[:-1])

      depth_image[:, :] = depth_arr[..., 0]

      rgb.append(channel_transform(rgb_arr)[:3, :, :])
      mask_responses.append(numpy.asarray([red_mask, blue_mask, green_mask]))
      depth_responses.append(numpy.clip(numpy.asarray([depth_image / 255.0]), 0, 1))
      normals_responses.append(channel_transform(normal_arr)[:3, :, :])
    yield (to_tensor(rgb, device=device),
           (to_tensor(mask_responses, device=device),
            to_tensor(depth_responses, device=device),
            to_tensor(normals_responses, device=device)
            ))


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

  terms = (dice,
           jaccard,
           ae_bce_loss,
           seg_bce_loss,
           depth_bce_loss,
           normals_bce_loss
           )

  term_weight = 1 / len(terms)
  weighted_terms = [term.mean() * term_weight for term in terms]

  loss = sum(weighted_terms)

  return NOD(loss=loss, terms=terms)
