#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import copy
import os
import time

from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from draugr.torch_utilities import reverse_channel_transform
from draugr.writers import TensorBoardPytorchWriter
from neodroid.wrappers import CameraObservationWrapper
from neodroidvision import PROJECT_APP_PATH
from neodroidvision.data.neodroid_environments.data import neodroid_camera_data_iterator, calculate_loss
from neodroidvision.segmentation import MultiHeadedSkipFCN

from neodroidvision.segmentation.segmentation_utilities import plot_utilities

__author__ = 'Christian Heider Nielsen'

import torch
from torch import optim
from tqdm import tqdm
from matplotlib import pyplot


def get_metric_str(metrics, writer, update_i):
  outputs = []
  for k, v in metrics:
    a = v.data.cpu().numpy()
    writer.add_scalar(f'loss/{k}', a, update_i)
    outputs.append(f'{k}:{a:2f}')

  return f'{", ".join(outputs)}'


def train_model(model,
                data_iterator,
                optimizer,
                scheduler,
                writer,
                interrupted_path,
                num_updates=25000):
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

        rgb_imgs, (seg_target,depth_target,normals_target) = next(data_iterator)

        optimizer.zero_grad()
        with torch.set_grad_enabled(phase == 'train'):
          recon_pred, seg_pred, depth_pred, normals_pred = model(rgb_imgs)
          ret = calculate_loss((seg_pred, seg_target),
                               (recon_pred, rgb_imgs),
                               (depth_pred, depth_target),
                               (normals_pred, normals_target)
                               )

          if phase == 'train':
            ret.loss.backward()
            optimizer.step()
            scheduler.step()

        update_loss = ret.loss.data.cpu().numpy()
        writer.scalar(f'loss/accum', update_loss, update_i)

        if phase == 'val' and update_loss < best_loss:
          best_loss = update_loss
          best_model_wts = copy.deepcopy(model.state_dict())
          format = 'NCHW'
          writer.image(f'rgb_imgs',
                       reverse_channel_transform(rgb_imgs),
                       update_i,
                       dataformats=format)
          writer.image(f'recon_pred', recon_pred, update_i, dataformats=format)
          writer.image(f'seg_target', seg_target, update_i, dataformats=format)
          writer.image(f'seg_pred', seg_pred, update_i, dataformats=format)
          writer.image(f'depth_target', depth_target, update_i, dataformats=format)
          writer.image(f'depth_pred', depth_pred, update_i, dataformats=format)
          writer.image(f'normals_pred', normals_pred, update_i, dataformats=format)
          writer.image(f'normals_target', normals_target, update_i, dataformats=format)
          sess.write(f'New best model at update {update_i}')

      # _ = get_metric_str(ret.terms, writer, update_i)
      sess.set_description_str(f'Update {update_i} - {phase} accum_loss:{update_loss:2f}')

      if update_loss < 0.1:
        break
  except KeyboardInterrupt:
    print('Interrupt')
  finally:
    model.load_state_dict(best_model_wts)  # load best model weights
    torch.save(model.state_dict(), interrupted_path)

  time_elapsed = time.time() - since
  print(f'{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
  print(f'Best val loss: {best_loss:3f}')

  return model


def inference(model, data_iterator, load_path=None):
  if load_path is not None:
    model.load_state_dict(torch.load(load_path))

  model.eval()

  inputs, (labels, _, _) = next(data_iterator)

  pred, recon, _, _ = model(inputs)
  pred = pred.data.cpu().numpy()
  recon = recon.data.cpu().numpy()
  l = labels.cpu().numpy()
  inputs = inputs.cpu().numpy()

  input_images_rgb = [reverse_channel_transform(x) for x in inputs]
  target_masks_rgb = [plot_utilities.masks_to_color_img(reverse_channel_transform(x)) for x in l]
  pred_rgb = [plot_utilities.masks_to_color_img(reverse_channel_transform(x)) for x in pred]
  pred_recon = [reverse_channel_transform(x) for x in recon]

  plot_utilities.plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb, pred_recon])
  pyplot.show()


def main():
  args = argparse.ArgumentParser()
  args.add_argument('-i', action='store_false')
  options = args.parse_args()

  seed = 42
  batch_size = 8  # 12
  depth = 3  # 5
  tqdm.monitor_interval = 0
  learning_rate = 3e-3
  lr_sch_step_size = int(1000 // batch_size) + 4
  lr_sch_gamma = 0.1
  model_start_channels = 16
  input_shape = (28,28)
  input_channels = 3

  home_path = PROJECT_APP_PATH
  best_model_path = 'INTERRUPTED_BEST.pth'
  interrupted_path = str(PROJECT_APP_PATH.user_data / best_model_path)

  env = CameraObservationWrapper()

  torch.manual_seed(seed)
  env.seed(seed)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  aeu_model = MultiHeadedSkipFCN(input_channels,
                                 (input_channels,3,1,3),
                                 encoding_depth=depth,
                                 start_channels=model_start_channels)
  aeu_model = aeu_model.to(device)

  optimizer_ft = optim.Adam(aeu_model.parameters(), lr=learning_rate)

  exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=lr_sch_step_size, gamma=lr_sch_gamma)

  data_iter = iter(neodroid_camera_data_iterator(env, device, batch_size))

  if options.i:
    with TensorBoardPytorchWriter(str(home_path.user_log / str(time.time()))) as writer:
      trained_aeu_model = train_model(aeu_model,
                                      data_iter,
                                      optimizer_ft,
                                      exp_lr_scheduler,
                                      writer,
                                      interrupted_path)
    inference(trained_aeu_model, data_iter)

  else:
    _list_of_files = home_path.glob('*')
    lastest_model_path = str(max(_list_of_files, key=os.path.getctime)) + f'/{best_model_path}'
    print('loading previous model: ' + lastest_model_path)
    inference(aeu_model, data_iter, load_path=lastest_model_path)

  torch.cuda.empty_cache()
  env.close()


if __name__ == '__main__':
  main()
