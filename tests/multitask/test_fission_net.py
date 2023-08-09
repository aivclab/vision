#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 29/10/2019
           """

import numpy
import torch
from draugr.torch_utilities import to_tensor

from neodroidvision.multitask.fission.skip_hourglass import (
    MergeMode,
    SkipHourglassFission,
)


def test_skip_fission_multi_dict():
    channels = 3
    model = SkipHourglassFission(
        input_channels=channels,
        output_heads={"RGB": channels, "Depth": 1},
        encoding_depth=2,
        merge_mode=MergeMode.Concat,
    )
    x = to_tensor(numpy.random.random((1, channels, 320, 320)), dtype=torch.float)
    out = model(x)
    loss = torch.sum(out["RGB"])
    loss.backward()
    if False:
        from matplotlib import pyplot

        im = out["RGB"].detach()
        print(im.shape)
        pyplot.imshow((torch.tanh(im[0].transpose(2, 0)) + 1) * 0.5)
        # pyplot.show()

        im2 = out["Depth"].detach()
        print(im2.shape)
        pyplot.imshow((torch.tanh(im2[0][0, :, :]) + 1) * 0.5)
        # pyplot.show()


def test_skip_fission_multi_int():
    channels = 3
    model = SkipHourglassFission(
        input_channels=channels,
        output_heads=(channels, 1),
        encoding_depth=2,
        merge_mode=MergeMode.Concat,
    )
    x = to_tensor(numpy.random.random((1, channels, 320, 320)), dtype=torch.float)
    out, out2, *_ = model(x)
    loss = torch.sum(out)
    loss.backward()
    if False:
        from matplotlib import pyplot

        im = out.detach()
        print(im.shape)
        pyplot.imshow((torch.tanh(im[0].transpose(2, 0)) + 1) * 0.5)
        # pyplot.show()

        im2 = out2.detach()
        print(im2.shape)
        pyplot.imshow((torch.tanh(im2[0][0, :, :]) + 1) * 0.5)
        # pyplot.show()


if __name__ == "__main__":
    test_skip_fission_multi_int()
    test_skip_fission_multi_dict()
