#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tqdm import tqdm

from neodroidvision import PROJECT_APP_PATH
from neodroidvision.multitask.fission_net.skip_hourglass import SkipHourglassFissionNet
from neodroidvision.segmentation import BCEDiceLoss, bool_dice
from pathlib import Path
from draugr.torch_utilities import torch_seed, global_torch_device
import cv2
from matplotlib import pyplot
import numpy
import pandas
import seaborn
import torch
from torch.utils.data import DataLoader

from neodroidvision.segmentation.segmentation_utilities.masks.run_length_encoding import \
  mask_to_run_length
from samples.segmentation.clouds.cloud_segmentation_utilities import (CloudDataset,
                                                                      post_process,
                                                                      resize_image_cv,
                                                                      visualize_with_raw,
                                                                      )

__author__ = 'Christian Heider Nielsen'
__doc__ = r'''

           Created on 09/10/2019
           '''


def reschedule(model, epoch, scheduler):
  "This can be improved its just a hacky way to write SGDWR "
  if epoch == 7:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    current_lr = next(iter(optimizer.param_groups))['lr']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           6,
                                                           eta_min=current_lr / 100,
                                                           last_epoch=-1)
  if epoch == 13:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    current_lr = next(iter(optimizer.param_groups))['lr']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           6,
                                                           eta_min=current_lr / 100,
                                                           last_epoch=-1)
  if epoch == 19:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002)
    current_lr = next(iter(optimizer.param_groups))['lr']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           6,
                                                           eta_min=current_lr / 100,
                                                           last_epoch=-1)
  if epoch == 25:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002)
    current_lr = next(iter(optimizer.param_groups))['lr']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           6,
                                                           eta_min=current_lr / 100,
                                                           last_epoch=-1)

  return model, scheduler


def train_d(model,
            train_loader,
            valid_loader,
            criterion,
            optimizer,
            scheduler,
            save_model_path,
            n_epochs=0):
  valid_loss_min = numpy.Inf  # track change in validation loss
  E = tqdm(range(1, n_epochs + 1))
  for epoch in E:
    train_loss = 0.0
    valid_loss = 0.0
    dice_score = 0.0

    model.train()
    train_set = tqdm(train_loader, postfix={"train_loss":0.0})
    for data, target in train_set:
      data, target = data.to(global_torch_device()), target.to(global_torch_device())
      optimizer.zero_grad()
      output, *_ = model(data)
      output = torch.sigmoid(output)
      loss = criterion(output, target)
      loss.backward()
      optimizer.step()
      train_loss += loss.item() * data.size(0)
      train_set.set_postfix(ordered_dict={"train_loss":loss.item()})

    model.eval()
    with torch.no_grad():
      validation_set = tqdm(valid_loader, postfix={"valid_loss":0.0, "dice_score":0.0})
      for data, target in validation_set:
        data, target = data.to(global_torch_device()), target.to(global_torch_device())
        # forward pass: compute predicted outputs by passing inputs to the model
        output, *_ = model(data)
        output = torch.sigmoid(output)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss
        valid_loss += loss.item() * data.size(0)
        dice_cof = bool_dice(output.cpu().detach().numpy(), target.cpu().detach().numpy())
        dice_score += dice_cof * data.size(0)
        validation_set.set_postfix(ordered_dict={"valid_loss":loss.item(), "dice_score":dice_cof})

    # calculate average losses
    train_loss = train_loss / len(train_loader.dataset)
    valid_loss = valid_loss / len(valid_loader.dataset)
    dice_score = dice_score / len(valid_loader.dataset)

    # print training/validation statistics
    E.set_description(f'Epoch: {epoch}'
                      f' Training Loss: {train_loss:.6f} '
                      f'Validation Loss: {valid_loss:.6f} '
                      f'Dice Score: {dice_score:.6f}')

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
      print(f'Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}).  Saving model ...')
      torch.save(model.state_dict(), save_model_path)
      valid_loss_min = valid_loss

    scheduler.step()
    model, scheduler = reschedule(model, epoch, scheduler)

  return model


def grid_search(model, probabilities, valid_masks, valid_loader):
  ## Grid Search for best Threshold
  class_params = {}

  for class_id in CloudDataset.classes.keys():
    print(CloudDataset.classes[class_id])
    attempts = []
    for t in range(0, 100, 5):
      t /= 100
      for ms in [0, 100, 1200, 5000, 10000, 30000]:
        masks, d = [], []
        for i in range(class_id, len(probabilities), 4):
          probability_ = probabilities[i]
          predict, num_predict = post_process(probability_, t, ms)
          masks.append(predict)
        for i, j in zip(masks, valid_masks[class_id::4]):
          if (i.sum() == 0) & (j.sum() == 0):
            d.append(1)
          else:
            d.append(bool_dice(i, j))
        attempts.append((t, ms, numpy.mean(d)))

    attempts_df = pandas.DataFrame(attempts, columns=['threshold', 'size', 'dice'])
    attempts_df = attempts_df.sort_values('dice', ascending=False)
    print(attempts_df.head())
    best_threshold = attempts_df['threshold'].values[0]
    best_size = attempts_df['size'].values[0]
    class_params[class_id] = (best_threshold, best_size)

  attempts_df = pandas.DataFrame(attempts, columns=['threshold', 'size', 'dice'])
  print(class_params)
  attempts_df.groupby(['threshold'])['dice'].max()

  attempts_df.groupby(['size'])['dice'].max()
  attempts_df = attempts_df.sort_values('dice', ascending=False)
  attempts_df.head(10)
  seaborn.lineplot(x='threshold', y='dice', hue='size', data=attempts_df)
  pyplot.title('Threshold and min size vs dice')
  best_threshold = attempts_df['threshold'].values[0]
  best_size = attempts_df['size'].values[0]

  for i, (data, target) in enumerate(valid_loader):
    data = data.to(global_torch_device())
    output, *_ = model(data)
    output = torch.sigmoid(output)[0].cpu().detach().numpy()
    image = data[0].cpu().detach().numpy()
    mask = target[0].cpu().detach().numpy()
    output = output.transpose(1, 2, 0)
    image_vis = image.transpose(1, 2, 0)
    mask = mask.astype('uint8').transpose(1, 2, 0)
    pr_mask = numpy.zeros((350, 525, 4))
    for j in range(4):
      probability_ = resize_image_cv(output[:, :, j])
      pr_mask[:, :, j], _ = post_process(probability_,
                                         class_params[j][0],
                                         class_params[j][1])
    visualize_with_raw(image=image_vis,
                       mask=pr_mask,
                       original_image=image_vis,
                       original_mask=mask,
                       raw_image=image_vis,
                       raw_mask=output)
    if i >= 6:
      break

  return class_params


def submission(model, class_params, base_path, batch_size, resized_loc):
  test_loader = DataLoader(CloudDataset(df_path=base_path / 'sample_submission.csv',
                                        resized_loc=resized_loc,
                                        subset='test'),
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=2)

  submit_ = pandas.read_csv(str(base_path / 'sample_submission.csv'))
  pathlist = [f'{base_path}/test_images/{i.split("_")[0]}' for i in submit_['Image_Label']]

  def get_black_mask(image_path):
    img = cv2.imread(image_path)
    img = resize_image_cv(img, (525, 350))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = numpy.array([0, 0, 0], numpy.uint8)
    upper = numpy.array([180, 255, 10], numpy.uint8)
    return (~ (cv2.inRange(hsv, lower, upper) > 250)).astype(int)

  pyplot.imshow(get_black_mask(pathlist[120]))
  pyplot.show()

  encoded_pixels = []
  image_id = 0
  cou = 0
  np_saved = 0
  for data, target in tqdm(test_loader):
    data = data.to(global_torch_device())
    output, *_ = model(data)
    output = torch.sigmoid(output)
    del data
    for i, batch in enumerate(output):
      for probability in batch:
        probability = resize_image_cv(probability.cpu().detach().numpy())
        predict, num_predict = post_process(probability,
                                            class_params[image_id % 4][0],
                                            class_params[image_id % 4][1])
        if num_predict == 0:
          encoded_pixels.append('')
        else:
          black_mask = get_black_mask(pathlist[cou])
          np_saved += numpy.sum(predict)
          predict = numpy.multiply(predict, black_mask)
          np_saved -= numpy.sum(predict)
          r = mask_to_run_length(predict)
          encoded_pixels.append(r)
        cou += 1
        image_id += 1

  print(f"number of pixel saved {np_saved}")

  submit_['EncodedPixels'] = encoded_pixels
  submit_.to_csv('submission.csv', columns=['Image_Label', 'EncodedPixels'], index=False)


def main():
  pyplot.style.use('bmh')

  base_path = Path.home() / 'Data' / 'Datasets' / 'Clouds'
  resized_loc = base_path / 'resized'

  save_model_path = PROJECT_APP_PATH.user_data / 'cloud_seg.model'

  SEED = 87539842
  batch_size = 8
  num_workers = 2
  torch_seed(SEED)

  min_size = (10000, 10000, 10000, 10000)

  train_loader = DataLoader(CloudDataset(df_path=base_path / 'train.csv',
                                         resized_loc=resized_loc,
                                         subset="train",
                                         ),
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers
                            )
  valid_loader = DataLoader(CloudDataset(df_path=base_path / 'train.csv',
                                         resized_loc=resized_loc,
                                         subset="valid",
                                         ),
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers
                            )

  model = SkipHourglassFissionNet(CloudDataset.predictors_shape[-1],
                                  CloudDataset.response_shape,
                                  encoding_depth=2)
  model.to(global_torch_device())

  if save_model_path.exists():
    model.load_state_dict(torch.load(str(save_model_path)))  # load last model

  criterion = BCEDiceLoss(eps=1.0)
  optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
  current_lr = next(iter(optimizer.param_groups))['lr']
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         7,
                                                         eta_min=current_lr / 100,
                                                         last_epoch=-1)
  model = train_d(model,
                  train_loader,
                  valid_loader,
                  criterion,
                  optimizer,
                  scheduler,
                  str(save_model_path))

  if save_model_path.exists():
    model.load_state_dict(torch.load(str(save_model_path)))  # load best model
  model.eval()

  # %%

  valid_masks = []
  count = 0
  tr = min(len(valid_loader.dataset) * 4, 2000)
  probabilities = numpy.zeros((tr, 350, 525), dtype=numpy.float32)
  for data, target in tqdm(valid_loader):
    data = data.to(global_torch_device())
    target = target.cpu().detach().numpy()
    outpu, *_ = model(data)
    outpu = torch.sigmoid(outpu).cpu().detach().numpy()
    for p in range(data.shape[0]):
      output, mask = outpu[p], target[p]
      for m in mask:
        valid_masks.append(resize_image_cv(m))
      for probability in output:
        probabilities[count, :, :] = resize_image_cv(probability)
        count += 1
      if count >= tr - 1:
        break
    if count >= tr - 1:
      break

  class_parameters = grid_search(model,
                                 probabilities,
                                 valid_masks,
                                 valid_loader)

  submission(model,
             class_parameters,
             base_path=base_path,
             batch_size=batch_size,
             resized_loc=resized_loc)


if __name__ == '__main__':
  main()
