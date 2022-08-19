from typing import Tuple

import torch
from torch import nn

__all__ = ["Compress"]


class Compress(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        pooling: bool = True,
        activation=torch.relu,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.activation = activation

        self.conv1 = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=1,
        )
        self.conv2 = nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=1,
        )

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
          x:

        Returns:

        """
        x = self.activation(self.conv2(self.activation(self.conv1(x))))
        before_pool = x

        if self.pooling:
            x = self.pool(x)

        return x, before_pool
