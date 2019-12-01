from typing import Any

import numpy
import torch
from torch import nn

from neodroidvision.segmentation.evaluation.f_score import f_score


def bool_dice(img1, img2):
  img1 = numpy.asarray(img1).astype(numpy.bool)
  img2 = numpy.asarray(img2).astype(numpy.bool)

  intersection = numpy.logical_and(img1, img2)

  return 2.0 * intersection.sum() / (img1.sum() + img2.sum())


def dice_coefficient(pred, target, *, epsilon=1e-10):
  """
  This definition generalize to real valued pred and target vector.
This should be differentiable.
  pred: tensor with first dimension as batch
  target: tensor with first dimension as batch
  """

  pred_flat = pred.contiguous().view(-1)  # have to use contiguous since they may from a torch.view op
  target_flat = target.contiguous().view(-1)

  intersection = 2. * (pred_flat * target_flat).sum() + epsilon
  union = (target_flat ** 2).sum() + (pred_flat ** 2).sum() + epsilon

  dice_coefficient = intersection / union

  return dice_coefficient


def dice_loss(prediction, target, *, epsilon=1e-10):
  return 1 - dice_coefficient(prediction, target, epsilon=epsilon)


class DiceLoss(nn.Module):
  __name__ = 'dice_loss'

  def __init__(self, *, eps: float = 1e-7, activation=torch.sigmoid):
    super().__init__()
    self.activation = activation
    self.eps = eps

  def forward(self, y_pr, y_gt):
    return 1 - f_score(y_pr,
                       y_gt,
                       beta=1.0,
                       eps=self.eps,
                       threshold=None,
                       activation=self.activation)


class BCEDiceLoss(DiceLoss):
  __name__ = 'bce_dice_loss'

  def __init__(self,
               eps: float = 1e-7,
               activation: Any = None,
               lambda_dice: float = 1.0,
               lambda_bce: float = 1.0):
    super().__init__(eps=eps, activation=activation)

    if activation == None:
      self.bce = nn.BCELoss(reduction='mean')
    else:
      self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    self.lambda_dice = lambda_dice
    self.lambda_bce = lambda_bce

  def forward(self, y_pr, y_gt):
    dice = super().forward(y_pr, y_gt)
    bce = self.bce(y_pr, y_gt)
    return (self.lambda_dice * dice) + (self.lambda_bce * bce)


if __name__ == '__main__':
  numpy.random.seed(2)
  data = numpy.random.random_sample((2, 1, 84, 84))
  LATEST_GPU_STATS = torch.FloatTensor(data)
  b = torch.FloatTensor(data.transpose((0, 1, 3, 2)))
  print(dice_loss(LATEST_GPU_STATS, LATEST_GPU_STATS))
  print(dice_loss(LATEST_GPU_STATS, b))

  h = torch.FloatTensor(numpy.array([[0, 1], [1, 1]]))
  j = torch.FloatTensor(numpy.ones((2, 2)))
  print(dice_loss(j, j))
