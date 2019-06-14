import numpy as np
import torch


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


if __name__ == '__main__':
  np.random.seed(2)
  data = np.random.random_sample((2, 1, 84, 84))
  LATEST_GPU_STATS = torch.FloatTensor(data)
  b = torch.FloatTensor(data.transpose((0, 1, 3, 2)))
  print(dice_loss(LATEST_GPU_STATS, LATEST_GPU_STATS))
  print(dice_loss(LATEST_GPU_STATS, b))

  h = torch.FloatTensor(np.array([[0, 1], [1, 1]]))
  j = torch.FloatTensor(np.ones((2, 2)))
  print(dice_loss(j, j))
