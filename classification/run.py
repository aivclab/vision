#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import copy
import os
import time
from pathlib import Path

import torchvision as torchvision
from PIL import Image
from warg.pooled_queue_processor import PooledQueueTask

from classification.data import (FileGenerator, NeodroidClassificationGenerator, a_retransform, a_transform)
from segmentation.segmentation_utilities import plot_utilities

__author__ = 'cnheider'

import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from neodroid.wrappers.observation_wrapper.observation_wrapper import CameraObservationWrapper


def get_metric_str(metrics, writer, update_i):
  outputs = []
  for k, v in metrics:
    a = v.data.cpu().numpy()
    writer.add_scalar(f'loss/{k}', a, update_i)
    outputs.append(f'{k}:{a:2f}')

  return f'{", ".join(outputs)}'


def train_model(model,
                data_iterator,
                criterion,
                optimizer,
                scheduler,
                writer,
                interrupted_path,
                num_updates=25000):
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 1e10
  since = time.time()

  try:
    sess = tqdm(range(num_updates), leave=False)
    for update_i in sess:
      for phase in ['train', 'val']:
        if phase == 'train':
          if scheduler:
            scheduler.step()
            for param_group in optimizer.param_groups:
              writer.add_scalar('lr', param_group['lr'], update_i)

          model.train()
        else:
          model.eval()

        rgb_imgs, true_label = next(data_iterator)

        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
          pred = model(rgb_imgs)
          loss = criterion(pred, true_label)

          if phase == 'train':
            loss.backward()
            optimizer.step()

        update_loss = loss.data.cpu().numpy()
        writer.add_scalar(f'loss/accum', update_loss, update_i)

        if phase == 'val' and update_loss < best_loss:
          best_loss = update_loss
          best_model_wts = copy.deepcopy(model.state_dict())
          writer.add_images(f'rgb_imgs', rgb_imgs, update_i)
          sess.write(f'New best model at update {update_i} with loss {best_loss}')
          torch.save(model.state_dict(), interrupted_path)

      sess.set_description_str(f'Update {update_i} - {phase} accum_loss:{update_loss:2f}')

      if update_loss < early_stop:
        break
  except KeyboardInterrupt:
    print('Interrupt')
  finally:
    pass

  model.load_state_dict(best_model_wts)  # load best model weights
  torch.save(model.state_dict(), interrupted_path)

  time_elapsed = time.time() - since
  print(f'{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
  print(f'Best val loss: {best_loss:3f}')

  return model

def set_parameter_requires_grad(model, feature_extracting):
  if feature_extracting:
    for param in model.parameters():
      param.requires_grad = False

def test_model(model, data_iterator, device='cpu', load_path=None):
  if load_path is not None:
    model.load_state_dict(torch.load(load_path))

  model.eval()

  inputs, labels = next(data_iterator)

  inputs = inputs.to(device)
  labels = labels.to(device)
  with torch.no_grad():
    pred = model(inputs)

  _, predicted = torch.max(pred, 1)[:6]
  pred = pred.data.cpu().numpy()[:6]
  l = labels.cpu().numpy()[:6]

  input_images_rgb = [a_retransform(x) for x in inputs][:6]

  plot_utilities.plot_prediction(input_images_rgb, l, predicted, pred)
  plt.show()


device = 'cpu'
seed = 42
batch_size = 8
tqdm.monitor_interval = 0
learning_rate = 3e-3
weight_decay = 0
lr_sch_step_size = 100
lr_sch_gamma = 0.1
num_classes = 4
momentum = 0.9
train_only_last_layer = True
early_stop=3e-6

env = CameraObservationWrapper()
env.seed(seed)


class FetchConvert(PooledQueueTask):
  def call(self, *args, **kwargs):
    predictors = []
    class_responses = []

    while len(predictors) < batch_size:
      state = env.update()
      rgb_arr = state.observer('RGB').observation_value
      rgb_arr = Image.open(rgb_arr).convert('RGB')
      a_class = state.observer('Class').observation_value

      predictors.append(a_transform(rgb_arr))
      class_responses.append(int(a_class))

    a = torch.stack(predictors).to(device)
    b = torch.LongTensor(class_responses).to(device)
    return (a, b)


def main():
  global device
  args = argparse.ArgumentParser()
  args.add_argument('--inference', '-i', action='store_true')
  args.add_argument('--continue_training', '-c', action='store_true')
  args.add_argument('--real_data', '-r', action='store_true')
  args.add_argument('--no_cuda', '-k', action='store_false')
  options = args.parse_args()

  home_path = Path.home() / 'Models' / 'Vision'
  base_path = home_path / str(time.time())
  best_model_path = 'best_test_model.pth'
  interrupted_path = str(base_path / best_model_path)

  torch.manual_seed(seed)

  if not options.no_cuda:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  #model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
  model = torchvision.models.resnet18(pretrained=True)
  set_parameter_requires_grad(model, train_only_last_layer)
  num_ftrs = model.fc.in_features
  model.fc = torch.nn.Linear(num_ftrs, num_classes)
  model = model.to(device)

  criterion = torch.nn.CrossEntropyLoss()

  params_to_update = model.parameters()
  print("Params to learn:")
  if train_only_last_layer:
    params_to_update = []
    for name, param in model.named_parameters():
      if param.requires_grad == True:
        params_to_update.append(param)
        print("\t", name)
  else:
    for name, param in model.named_parameters():
      if param.requires_grad == True:
        print("\t", name)

  # optimizer_ft = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
  optimizer_ft = optim.SGD(params_to_update,
                           lr=learning_rate,
                           momentum=momentum
                           # , weight_decay=weight_decay
                           )
  # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=lr_sch_step_size, gamma=lr_sch_gamma)
  exp_lr_scheduler = None

  if options.real_data:
    data_iter = iter(FileGenerator(path='/home/heider/Data/Datasets/Vision/vestas/real/train'))
  else:
    data_iter = iter(NeodroidClassificationGenerator(env, device, batch_size))
    # data_iter = iter(NeodroidClassificationGenerator(FetchConvert()))

  if options.continue_training:
    _list_of_files = home_path.glob('*')
    latest_model_path = str(max(_list_of_files, key=os.path.getctime)) + f'/{best_model_path}'
    print('loading previous model: ' + latest_model_path)
    if latest_model_path is not None:
      model.load_state_dict(torch.load(latest_model_path))

  if not options.inference:
    writer = SummaryWriter(str(base_path))
    # writer.add_graph(model)
    trained_model = train_model(model,
                                data_iter,
                                criterion,
                                optimizer_ft,
                                exp_lr_scheduler,
                                writer,
                                interrupted_path)
    test_model(trained_model, data_iter, device=device)
    writer.close()
  else:
    _list_of_files = home_path.glob('*')
    latest_model_path = str(max(_list_of_files, key=os.path.getctime)) + f'/{best_model_path}'
    print('loading previous model: ' + latest_model_path)
    test_model(model, data_iter, load_path=latest_model_path, device=device)

  torch.cuda.empty_cache()
  if not options.real_data:
    env.close()


if __name__ == '__main__':
  main()
