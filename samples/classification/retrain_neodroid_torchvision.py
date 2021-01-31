#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import time

import torchvision
from matplotlib import pyplot
from neodroid.wrappers.observation_wrapper.mixed_observation_wrapper import (
  MixedObservationWrapper,
  )
from neodroidvision import PROJECT_APP_PATH
from neodroidvision.classification import (
  pred_target_train_model,
  squeezenet_retrain,
  )

from draugr import batch_generator, horizontal_imshow
from draugr.python_utilities import (
  rgb_drop_alpha_batch_nhwc,
  torch_vision_normalize_batch_nchw,
  )
from draugr.torch_utilities import (
  TensorBoardPytorchWriter,
  TorchEvalSession,
  ensure_directory_exist,
  global_torch_device,
  to_tensor,
  uint_nhwc_to_nchw_float_batch,
  )

# from warg.pooled_queue_processor import PooledQueueTask

__author__ = "Christian Heider Nielsen"

import torch
import torch.optim as optim
from tqdm import tqdm

seed = 34874312
batch_size = 16
tqdm.monitor_interval = 0
learning_rate = 3e-5
momentum = 0.9
wd = 3e-8
test_batch_size = batch_size
early_stop = 3e-6
num_updates = 6000
lr_cycles = 1
flatt_size = 224 * 224 * 3

normalise = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )


def main():
  args = argparse.ArgumentParser()
  args.add_argument("--inference", "-i", action="store_true")
  args.add_argument("--continue_training", "-c", action="store_true")
  args.add_argument("--no_cuda", "-k", action="store_false")
  args.add_argument("--export", "-e", action="store_true")
  options = args.parse_args()

  timeas = str(time.time())
  this_model_path = PROJECT_APP_PATH.user_data / timeas
  this_log = PROJECT_APP_PATH.user_log / timeas
  ensure_directory_exist(this_model_path)
  ensure_directory_exist(this_log)

  best_model_name = "best_validation_model.model"
  interrupted_path = str(this_model_path / best_model_name)

  torch.manual_seed(seed)

  if not options.no_cuda:
    global_torch_device("cpu")

  with MixedObservationWrapper() as env:
    env.seed(seed)
    train_iter = batch_generator(iter(env), batch_size)
    num_categories = env.sensor("Class").space.discrete_steps
    test_iter = train_iter

    model, params_to_update = squeezenet_retrain(num_categories)
    print(params_to_update)

    model = model.to(global_torch_device())

    if options.continue_training:
      _list_of_files = list(PROJECT_APP_PATH.user_data.rglob("*.model"))
      lastest_model_path = str(max(_list_of_files, key=os.path.getctime))
      print(f"loading previous model: {lastest_model_path}")
      if lastest_model_path is not None:
        model.load_state_dict(torch.load(lastest_model_path))

    criterion = torch.nn.CrossEntropyLoss().to(global_torch_device())

    optimizer_ft = optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=wd
        )
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1
        )

    with TensorBoardPytorchWriter(this_log) as writer:
      if True:
        model = pred_target_train_model(
            model,
            train_iter,
            criterion,
            optimizer_ft,
            exp_lr_scheduler,
            writer,
            interrupted_path,
            test_data_iterator=test_iter,
            num_updates=num_updates,
            )

      inputs, true_label = zip(*next(train_iter))
      rgb_imgs = torch_vision_normalize_batch_nchw(
          uint_nhwc_to_nchw_float_batch(
              rgb_drop_alpha_batch_nhwc(to_tensor(inputs))
              )
          )

      pred = model(rgb_imgs)
      predicted = torch.argmax(pred, -1)
      true_label = to_tensor(true_label, dtype=torch.long)
      print(predicted, true_label)
      horizontal_imshow(
          inputs,
          [f"p:{int(p)},t:{int(t)}" for p, t in zip(predicted, true_label)],
          )
      pyplot.show()

  torch.cuda.empty_cache()

  if options.export:
    with TorchEvalSession(model):
      example = torch.rand(1, 3, 256, 256)
      traced_script_module = torch.jit.trace(model.to("cpu"), example)
      traced_script_module.save("resnet18_v.model")


if __name__ == "__main__":
  main()
