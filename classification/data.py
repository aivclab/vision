#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from multiprocessing import Pipe, Process


import torch.nn.functional as F
from PIL import Image
from neodroid.neodroid_utilities.encodings import to_one_hot
from neodroid.wrappers.observation_wrapper.observation_wrapper import CameraObservationWrapper
from torch.utils.data import Dataset
from warg import NOD
import numpy as np

from segmentation.segmentation_utilities.plot_utilities import channel_transform

__author__ = 'cnheider'

import torch


def NeodroidClassificationGenerator(env, device, batch_size=64):
  while True:
    predictors = []
    class_responses = []
    while len(predictors) < batch_size:
      state = env.update()
      rgb_arr = state.observer('RGB').observation_value
      rgb_arr = np.asarray(Image.open(rgb_arr).convert('RGB'))
      a_class = state.observer('Class').observation_value


      predictors.append(channel_transform(rgb_arr))
      #class_responses.append(to_one_hot(4, int(a_class)))
      class_responses.append(int(a_class))
    yield torch.FloatTensor(predictors).to(device), torch.LongTensor(class_responses).to(device)




class NeodroidDataGenerator(Dataset):

  @staticmethod
  def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
      cmd, data = remote.recv()
      if cmd == 'step':
        ob, signal, terminal, info = env.step(data)
        if terminal:
          ob = env.reset()
        remote.send((ob, signal, terminal, info))

  def __init__(self,
               *,
               connect_to_running=True,
               env_name='',
               max_buffer_size=255,
               generation_workers=1,
               transformation_workers=1):
    self._max_buffer_size = max_buffer_size
    self._generation_workers = generation_workers
    self._transformation_workers = transformation_workers
    self._connect_to_running = connect_to_running
    self._env_name = env_name

    self._env = CameraObservationWrapper(connect_to_running=self._connect_to_running,
                                         env_name=self._env_name)

  def start_async(self):
    pass


  def start_workers(self):
    for _ in range(self._generation_workers):
      self._env = CameraObservationWrapper(connect_to_running=self._connect_to_running,
                                           env_name=self._env_name)

    self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self._transformation_workers)])
    self.ps = [Process(target=self.worker, args=(work_remote,
                                                 remote,
                                                 CloudpickleWrapper(env_fn)))
               for (work_remote, remote, env_fn) in zip(self.work_remotes,
                                                        self.remotes,
                                                        env_fns)]
    for p in self.ps:
      p.daemon = True  # if the main process crashes, we should not cause things to hang
      p.start()

  def __getitem__(self, index):
    state = self._env.update()
    rgb_arr = state.observer('RGB').observation_value
    rgb_arr = np.asarray(Image.open(rgb_arr).convert('RGB'))
    a_class = state.observer('Class').observation_value

    predictors = channel_transform(rgb_arr)
    #class_responses = to_one_hot(4, int(a_class))
    class_responses = int(a_class)
    return predictors, class_responses

  def __len__(self):
    return self._max_buffer_size


if __name__ == '__main__':
  neodroid_generator = NeodroidDataGenerator()
  train_loader = torch.utils.data.DataLoader(dataset=neodroid_generator,
                                             batch_size=12,
                                             shuffle=True)
  for p, r in train_loader:
    print(r)
