#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import time
from pathlib import Path

from torch import nn

from vision.classification import (FileGenerator,
                                   NeodroidClassificationGenerator,
                                   export,
                                   squeezenet_retrain,
                                   test_model,
                                   train_model,
                                   )

# from warg.pooled_queue_processor import PooledQueueTask

__author__ = 'cnheider'

import torch
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
from neodroid.wrappers.observation_wrapper.observation_wrapper import CameraObservationWrapper

device = 'cpu'
seed = 42
batch_size = 16
tqdm.monitor_interval = 0
learning_rate = 3e-3
weight_decay = 0
lr_sch_step_size = 1000
lr_sch_gamma = 0.1
num_classes = 4
momentum = 0.9
test_batch_size = batch_size
early_stop = 3e-6

# real_data_path = Path.home() / 'Data' / 'Datasets' / 'Classification' / 'vestas' / 'real' / 'all'
real_data_path = Path.home() / 'Data' / 'Datasets' / 'Classification' / 'vestas' / 'real' / 'val'
models_path = Path.home() / 'Models' / 'Vision'
this_model_path = models_path / str(time.time())


def main():
  global device
  args = argparse.ArgumentParser()
  args.add_argument('--inference', '-i', action='store_true')
  args.add_argument('--continue_training', '-c', action='store_true')
  args.add_argument('--real_data', '-r', action='store_true')
  args.add_argument('--no_cuda', '-k', action='store_false')
  args.add_argument('--export', '-e', action='store_true')
  options = args.parse_args()

  best_model_name = 'best_validation_model.pth'
  interrupted_path = str(this_model_path / best_model_name)

  torch.manual_seed(seed)

  if not options.no_cuda:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  model, params_to_update = squeezenet_retrain(num_classes)
  # model, params_to_update = resnet_retrain(num_classes,resnet_version=torchvision.models.resnet50)

  model = model.to(device)

  criterion = torch.nn.CrossEntropyLoss()
  # criterion = torch.nn.NLLLoss()

  optimizer_ft = optim.Adam(params_to_update, lr=learning_rate, weight_decay=weight_decay)
  exp_lr_scheduler = None

  test_data_iter = FileGenerator(path=real_data_path, batch_size=test_batch_size)()

  _list_of_files = models_path.glob('*')
  latest = str(max(_list_of_files, key=os.path.getctime))
  latest_model_path = latest + f'/{best_model_name}'
  model_export_name = latest + f'/classification.onnx'
  if options.continue_training:
    if latest_model_path is not None:
      print('loading previous model: ' + latest_model_path)
      model.load_state_dict(torch.load(latest_model_path))

  if not options.inference:
    env = CameraObservationWrapper()
    env.seed(seed)
    data_iter = iter(NeodroidClassificationGenerator(env, device, batch_size))
    # data_iter = iter(NeodroidClassificationGenerator2())
    writer = SummaryWriter(str(this_model_path))
    # writer.add_graph(model)
    trained_model = train_model(model,
                                data_iter,
                                test_data_iter,
                                criterion,
                                optimizer_ft,
                                exp_lr_scheduler,
                                writer,
                                interrupted_path,
                                device=device)
    test_model(trained_model, test_data_iter, latest_model_path)
    writer.close()
    env.close()
  else:
    if latest_model_path is not None:
      print('loading previous model: ' + latest_model_path)
      model.load_state_dict(torch.load(latest_model_path))
    test_model(model, test_data_iter, latest_model_path)

  torch.cuda.empty_cache()

  if options.export:
    export(model, model_export_name, latest)


if __name__ == '__main__':
  main()
