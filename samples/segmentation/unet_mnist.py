#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import copy
import os
import time
from itertools import cycle
from pathlib import Path
from typing import Iterator

from torch.nn.modules.module import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from draugr import global_torch_device
from draugr.writers import TensorBoardPytorchWriter, ImageWriter
from neodroidvision import PROJECT_APP_PATH
from neodroidvision.data.to_device_tensor_iterator import to_device_tensor_iterator
from neodroidvision.segmentation import SkipBottleNeckFissionNet
from neodroidvision.utilities.torch_utilities import MinMaxNorm

__author__ = 'Christian Heider Nielsen'

import torch
from torch import optim
from tqdm import tqdm
from torchvision.datasets import MNIST

criterion = torch.nn.MSELoss()


def training(model: Module,
             data_iterator: Iterator,
             optimizer: Optimizer,
             scheduler,
             writer: ImageWriter,
             interrupted_path: Path,
             num_updates=2500000,
             early_stop_threshold=1e-9) -> Module:
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 1e10
  since = time.time()

  try:
    sess = tqdm(range(num_updates), leave=False, disable=False)
    for update_i in sess:
      for phase in ['train', 'val']:
        if phase == 'train':

          for param_group in optimizer.param_groups:
            writer.scalar('lr', param_group['lr'], update_i)

          model.train()
        else:
          model.eval()

        rgb_imgs, *_ = next(data_iterator)

        optimizer.zero_grad()
        with torch.set_grad_enabled(phase == 'train'):
          recon_pred, *_ = model(rgb_imgs)
          ret = criterion(recon_pred, rgb_imgs)

          if phase == 'train':
            ret.backward()
            optimizer.step()
            scheduler.step()

        update_loss = ret.data.cpu().numpy()
        writer.scalar(f'loss/accum', update_loss, update_i)

        if phase == 'val' and update_loss < best_loss:
          best_loss = update_loss
          best_model_wts = copy.deepcopy(model.state_dict())
          _format = 'NCHW'
          writer.image(f'rgb_imgs',
                       rgb_imgs,
                       update_i,
                       dataformats=_format)
          writer.image(f'recon_pred', recon_pred, update_i, dataformats=_format)
          sess.write(f'New best model at update {update_i}')

      sess.set_description_str(f'Update {update_i} - {phase} accum_loss:{update_loss:2f}')

      if update_loss < early_stop_threshold:
        break
  except KeyboardInterrupt:
    print('Interrupt')
  finally:
    model.load_state_dict(best_model_wts)  # load best model weights
    torch.save(model.state_dict(), interrupted_path)

  time_elapsed = time.time() - since
  print(f'{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
  print(f'Best val loss: {best_loss}')

  return model


def inference(model: Module, data_iterator: Iterator):
  model.eval()
  inputs, *_ = next(data_iterator)
  pred, *_ = model(inputs)


def main(load_earlier=False, train=True):
  seed = 2554215
  batch_size = 32

  tqdm.monitor_interval = 0
  learning_rate = 3e-3
  lr_sch_step_size = int(10e4 // batch_size)
  lr_sch_gamma = 0.1
  unet_depth = 3
  unet_start_channels = 16
  input_channels = 1
  output_channels = (input_channels,)

  home_path = PROJECT_APP_PATH
  model_file_ending = '.model'
  interrupted_path = PROJECT_APP_PATH.user_data / f'INTERRUPTED_BEST{model_file_ending}'

  torch.manual_seed(seed)

  device = global_torch_device()

  img_transform = transforms.Compose([transforms.ToTensor(),
                                      MinMaxNorm(),
                                      transforms.Lambda(lambda tensor:torch.round(tensor))
                                      ])
  dataset = MNIST(PROJECT_APP_PATH.user_data / 'mnist',
                  transform=img_transform,
                  download=True)
  data_iter = iter(cycle(DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True)))
  data_iter = to_device_tensor_iterator(data_iter, device)

  model = SkipBottleNeckFissionNet(input_channels,
                                   output_channels,
                                   encoding_depth=unet_depth,
                                   start_channels=unet_start_channels).to(get_torch_device())

  optimizer_ft = optim.Adam(model.parameters(), lr=learning_rate)

  exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft,
                                               step_size=lr_sch_step_size,
                                               gamma=lr_sch_gamma)

  if load_earlier:
    _list_of_files = PROJECT_APP_PATH.user_data.glob('*.model')
    lastest_model_path = str(max(_list_of_files, key=os.path.getctime))
    print(f'loading previous model: {lastest_model_path}')
    if lastest_model_path is not None:
      model.load_state_dict(torch.load(lastest_model_path))

  if train:
    with TensorBoardPytorchWriter(home_path.user_log / str(time.time())) as writer:
      model = training(model,
                       data_iter,
                       optimizer_ft,
                       exp_lr_scheduler,
                       writer,
                       interrupted_path)
      torch.save(model, PROJECT_APP_PATH.user_data / f'fission_net{model_file_ending}')

  inference(model, data_iter)

  torch.cuda.empty_cache()


if __name__ == '__main__':
  main()
