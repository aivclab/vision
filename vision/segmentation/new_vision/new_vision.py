#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from PIL import Image

from vision.segmentation.new_vision.new_vision_engine import evaluate, train_one_epoch
from vision.segmentation.new_vision.new_vision_transforms import Compose, RandomHorizontalFlip, ToTensor
from vision.segmentation.new_vision.new_vision_utils import PennFudanDataset, collate_fn

__author__ = 'cnheider'
__doc__ = ''

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_instance_segmentation_model(num_classes):
  # load an instance segmentation model pre-trained on COCO
  model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

  # get the number of input features for the classifier
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  # replace the pre-trained head with a new one
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

  # now get the number of input features for the mask classifier
  in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
  hidden_layer = 256
  # and replace the mask predictor with a new one
  model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                     hidden_layer,
                                                     num_classes)

  return model


def get_transform(train):
  transforms = [ToTensor()]
  # converts the image, a PIL image, into a PyTorch Tensor
  if train:
    # during training, randomly flip the training images
    # and ground-truth for data augmentation
    transforms.append(RandomHorizontalFlip(0.5))
  return Compose(transforms)


if __name__ == '__main__':

  dataset = PennFudanDataset('/home/heider/Data/Datasets/Segmentation/PennFudanPed/',
                             get_transform(train=True))
  dataset_test = PennFudanDataset('/home/heider/Data/Datasets/Segmentation/PennFudanPed/', get_transform(
      train=False))

  # split the dataset in train and test set
  torch.manual_seed(1)
  indices = torch.randperm(len(dataset)).tolist()
  dataset = torch.utils.data.Subset(dataset, indices[:-50])
  dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

  # define training and validation data loaders
  data_loader = torch.utils.data.DataLoader(
      dataset, batch_size=2, shuffle=True, num_workers=4,
      collate_fn=collate_fn)

  data_loader_test = torch.utils.data.DataLoader(
      dataset_test, batch_size=1, shuffle=False, num_workers=4,
      collate_fn=collate_fn)

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  # our dataset has two classes only - background and person
  num_classes = 2

  # get the model using our helper function
  model = get_instance_segmentation_model(num_classes)
  # move model to the right device
  model.to(device)

  # construct an optimizer
  params = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.SGD(params, lr=0.005,
                              momentum=0.9, weight_decay=0.0005)

  # and a learning rate scheduler which decreases the learning rate by
  # 10x every 3 epochs
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                 step_size=3,
                                                 gamma=0.1)

  num_epochs = 10

  for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

  # pick one image from the test set
  img, _ = dataset_test[0]
  # put the model in evaluation mode
  model.eval()
  with torch.no_grad():
    prediction = model([img.to(device)])

  Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

  Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())
