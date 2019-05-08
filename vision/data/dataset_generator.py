#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import imageio
import numpy

__author__ = 'cnheider'
__doc__ = ''

from tqdm import tqdm

tqdm.monitor_interval = 0

from neodroid.wrappers import CameraObservationWrapper


def main():
  _environment = CameraObservationWrapper(connect_to_running=True, image_size=(512, 512, 4))

  frame_i = 0
  how_many = 100
  _environment.reset()

  observation_session = tqdm(_environment, leave=False)
  for obs in observation_session:
    frame_i += 1

    if frame_i % how_many == 0:
      observation_session.set_description(f'Frame: {frame_i}')
      break

    rgb = obs['RGB']
    obj = obs['ObjectSpace']

    name = f'stepper_{frame_i}'

    imageio.imwrite(name + '.png',rgb)

    numpy.savez_compressed(name + '.npz', obj.astype(numpy.float16))

    # with open(name+'.pk', 'wb') as f:

    # pickle.dump(obs,                  f,                  protocol=pickle.HIGHEST_PROTOCOL)

    # with open(f'data_{frame_i}.pk', 'rb') as f:
    #  o = pickle.load(f)

    #  print(o == obs)
    # print(obs)

  _environment.close()


if __name__ == '__main__':
  main()
