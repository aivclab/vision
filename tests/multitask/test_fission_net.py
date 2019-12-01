#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
import torch

from neodroidvision.multitask.fission_net.skip_hourglass import SkipHourglassFissionNet

__author__ = 'Christian Heider Nielsen'
__doc__ = r'''

           Created on 29/10/2019
           '''


def test_skip_fission_multi_dict():
  channels = 3
  model = SkipHourglassFissionNet(input_channels=channels, output_channels={'RGB':channels, 'Depth':1},
                                  encoding_depth=2,
                                  merge_mode='concat')
  x = torch.FloatTensor(numpy.random.random((1, channels, 320, 320)))
  out = model(x)
  loss = torch.sum(out['RGB'])
  loss.backward()
  from matplotlib import pyplot

  im = out['RGB'].detach()
  print(im.shape)
  pyplot.imshow((torch.tanh(im[0].transpose(2, 0)) + 1) * 0.5)
  pyplot.show()

  im2 = out['Depth'].detach()
  print(im2.shape)
  pyplot.imshow((torch.tanh(im2[0][0, :, :]) + 1) * 0.5)
  pyplot.show()


def test_skip_fission_multi_int():
  channels = 3
  model = SkipHourglassFissionNet(input_channels=channels, output_channels=(channels, 1), encoding_depth=2,
                                  merge_mode='concat')
  x = torch.FloatTensor(numpy.random.random((1, channels, 320, 320)))
  out, out2, *_ = model(x)
  loss = torch.sum(out)
  loss.backward()
  from matplotlib import pyplot

  im = out.detach()
  print(im.shape)
  pyplot.imshow((torch.tanh(im[0].transpose(2, 0)) + 1) * 0.5)
  pyplot.show()

  im2 = out2.detach()
  print(im2.shape)
  pyplot.imshow((torch.tanh(im2[0][0, :, :]) + 1) * 0.5)
  pyplot.show()
