import torch
import torch.nn as nn


class Compress(nn.Module):
  """
  A helper Module that performs 2 convolutions and 1 MaxPool.
  A ReLU activation follows each convolution.
  """

  def __init__(self, in_channels, out_channels, pooling=True):
    super().__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.pooling = pooling

    self.conv1 = nn.Conv2d(self.in_channels, self.out_channels,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           bias=True,
                           groups=1)
    self.conv2 = nn.Conv2d(self.out_channels,
                           self.out_channels,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           bias=True,
                           groups=1)

    if self.pooling:
      self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

  def forward(self, x):
    x = torch.relu(self.conv1(x))
    x = torch.relu(self.conv2(x))
    before_pool = x

    if self.pooling:
      x = self.pool(x)

    return x, before_pool
