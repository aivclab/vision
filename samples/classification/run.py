#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import time

from draugr import (TensorBoardPytorchWriter,
                    ensure_directory_exist,
                    generator_batch,
                    get_global_torch_device,
                    )
from neodroid.wrappers.observation_wrapper.mixed_observation_wrapper import MixedObservationWrapper
from neodroidvision import PROJECT_APP_PATH
from neodroidvision.classification import (pred_target_train_model, resnet_retrain)

# from warg.pooled_queue_processor import PooledQueueTask

__author__ = 'Christian Heider Nielsen'

import torch
import torch.optim as optim
from tqdm import tqdm

DEVICE = 'cpu'
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


# real_data_path = Path.home() / 'Data' / 'Datasets' / 'Classification' / 'vestas' / 'real' / 'val'


def main():
  global DEVICE
  args = argparse.ArgumentParser()
  args.add_argument('--inference', '-i', action='store_true')
  args.add_argument('--continue_training', '-c', action='store_false')
  args.add_argument('--real_data', '-r', action='store_true')
  args.add_argument('--no_cuda', '-k', action='store_false')
  args.add_argument('--export', '-e', action='store_true')
  options = args.parse_args()

  timeas = str(time.time())
  this_model_path = PROJECT_APP_PATH.user_data / timeas
  this_log = PROJECT_APP_PATH.user_log / timeas
  ensure_directory_exist(this_model_path)
  ensure_directory_exist(this_log)

  best_model_name = 'best_validation_model.model'
  interrupted_path = str(this_model_path / best_model_name)

  torch.manual_seed(seed)

  if not options.no_cuda:
    DEVICE = get_global_torch_device()

  model, params_to_update = resnet_retrain(num_classes)
  model = model.to(get_global_torch_device())

  if options.continue_training:
    _list_of_files = PROJECT_APP_PATH.user_data.rglob('*.model')
    lastest_model_path = str(max(_list_of_files, key=os.path.getctime))
    print(f'loading previous model: {lastest_model_path}')
    if lastest_model_path is not None:
      model.load_state_dict(torch.load(lastest_model_path))

  criterion = torch.nn.CrossEntropyLoss()

  optimizer_ft = optim.Adam(params_to_update, lr=learning_rate, weight_decay=weight_decay)
  exp_lr_scheduler = None

  env = MixedObservationWrapper()
  env.seed(seed)
  data_iter = generator_batch(iter(env), batch_size)

  writer = TensorBoardPytorchWriter(this_log)

  trained_model = pred_target_train_model(model,
                                          data_iter,
                                          criterion,
                                          optimizer_ft,
                                          exp_lr_scheduler,
                                          writer,
                                          interrupted_path,
                                          device=DEVICE)

  writer.close()
  env.close()

  torch.cuda.empty_cache()


if __name__ == '__main__':
  main()
