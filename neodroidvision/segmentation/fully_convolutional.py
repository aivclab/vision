#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 18/07/2020
           """

import math

import torch
from torch import nn
from torch.optim import Adam

from neodroidvision.segmentation import dice_loss

__all__ = ["FullyConvolutional", "FCN"]


class FullyConvolutional(nn.Module):
    """ """

    @staticmethod
    def _pad(kernel_size: int, stride: int, dilation: int = 1) -> int:
        """
        if length % stride == 0:
        out_length = length // stride
        else:
        out_length = length // stride + 1

        return math.ceil((out_length * stride + kernel_size - length - stride) / 2)"""
        return math.ceil((1 - stride + dilation * (kernel_size - 1)) / 2)

    @staticmethod
    def conv2d_pool_block(
        in_channels: int, out_channels: int, ext: bool = False
    ) -> torch.nn.Module:
        """

        Args:
          in_channels:
          out_channels:
          ext:

        Returns:

        """
        base_c = [
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=FullyConvolutional._pad(3, 1),
            ),
            torch.nn.ELU(),
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=FullyConvolutional._pad(3, 1),
            ),
            torch.nn.ELU(),
        ]
        if ext:
            base_c.extend(
                [
                    torch.nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=FullyConvolutional._pad(3, 1),
                    ),
                    torch.nn.ELU(),
                ]
            )
        base_c.append(
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )  # Valid padding
        return torch.nn.Sequential(*base_c)

    def __init__(
        self,
        in_channels: int,
        num_categories: int,
        *,
        final_act: callable,
        base: int = 4,
        t=8,
    ):
        """
        FCN8

        :param num_categories:
        :type num_categories:
        :param base:
        :type base:"""

        super().__init__()
        i_c = in_channels
        for ith_block in (0, 1):
            i_c_n = 2 ** (base + ith_block)
            setattr(self, f"pool_block{ith_block}", self.conv2d_pool_block(i_c, i_c_n))
            i_c = i_c_n

        for ith_block in (2, 3):
            i_c_n = 2 ** (base + ith_block)
            setattr(
                self,
                f"pool_block{ith_block}",
                self.conv2d_pool_block(i_c, i_c_n, ext=True),
            )
            i_c = i_c_n

        self.pool_block4 = self.conv2d_pool_block(i_c, 2 ** (base + 3), ext=True)

        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(
                i_c, 2048, kernel_size=7, padding=FullyConvolutional._pad(7, 1)
            ),
            torch.nn.Dropout(0.5),
        )
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(
                2048, 2048, kernel_size=1, padding=FullyConvolutional._pad(1, 1)
            ),
            torch.nn.Dropout(0.5),
        )

        for ith_block, ic2 in zip((2, 3), (num_categories, 2048)):
            i_c_n = 2 ** (base + ith_block)
            setattr(
                self,
                f"skip_block{ith_block}",
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        i_c_n,
                        num_categories,
                        kernel_size=1,
                        padding=FullyConvolutional._pad(1, 1),
                    ),
                    torch.nn.ELU(),
                ),
            )
            setattr(
                self,
                f"transpose_block{ith_block}",
                torch.nn.ConvTranspose2d(
                    ic2,
                    num_categories,
                    kernel_size=2,
                    stride=2,
                    padding=FullyConvolutional._pad(2, 2),
                ),
            )

        self.head = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                num_categories,
                num_categories,
                kernel_size=8,
                stride=8,
                padding=FullyConvolutional._pad(8, 8),
            ),
            final_act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
          x:

        Returns:

        """
        for ith_block in (0, 1):
            x = getattr(self, f"pool_block{ith_block}")(x)

        pool2 = self.pool_block2(x)
        pool3 = self.pool_block3(pool2)
        x = self.conv6(self.conv5(self.pool_block4(pool3)))

        s1, t1 = self.skip_block3(pool3), self.transpose_block3(x)
        print(s1.shape, t1.shape)
        x = s1 + t1
        x = self.skip_block2(pool2) + self.transpose_block2(x)

        return self.head(x)


FCN = FullyConvolutional

if __name__ == "__main__":

    def a():
        """ """
        img_size = 224
        in_channels = 5
        n_classes = 2
        metrics = dice_loss

        if n_classes == 1:
            # loss = 'binary_crossentropy'
            loss = torch.nn.BCELoss()
            final_act = torch.nn.Sigmoid()
        elif n_classes > 1:
            # loss = 'categorical_crossentropy'
            loss = torch.nn.CrossEntropyLoss()
            final_act = torch.nn.LogSoftmax(1)  # across channels

        model = FCN(in_channels, n_classes, final_act=final_act)
        optimiser = Adam(model.parameters(), 1e-4)

        pred = model(torch.ones((4, in_channels, img_size, img_size)))
        print(pred)

    a()
