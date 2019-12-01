#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import OpenEXR as exr
import Imath
import imageio

__author__ = 'Christian Heider Nielsen'
__doc__ = r'''

           Created on 09/10/2019
           '''


def read_exr_file(filename):
  """Read RGB + Depth data from EXR image file.
  Parameters
  ----------
  filename : str
      File path.
  Returns
  -------
  img : RGB image in float32 format.
  Z : Depth buffer in float3.
  """

  exrfile = exr.InputFile(filename)
  dw = exrfile.header()['dataWindow']
  isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

  channels = ['R', 'G', 'B', 'Z']
  channelData = dict()

  for c in channels:
    C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
    C = np.fromstring(C, dtype=np.float32)
    C = np.reshape(C, isize)

    channelData[c] = C

  # create RGB image
  img = np.concatenate([channelData[c][..., np.newaxis] for c in ['R', 'G', 'B']], axis=2)

  return img, channelData['Z']


print(read_exr_file(str(Path.home() / 'Downloads' / 'untitled.exr'))[0])
