#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import albumentations
import numpy
import pandas
from matplotlib import pyplot
from pathlib import Path
import cv2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

from neodroidvision.segmentation.segmentation_utilities.masks.drawing import draw_convex_hull
from neodroidvision.segmentation.segmentation_utilities.masks.run_length_encoding import \
  run_length_encoding2mask

__author__ = 'Christian Heider Nielsen'
__doc__ = r'''
           Created on 20/10/2019
           '''


class CloudDataset(Dataset):
  @staticmethod
  def get_training_augmentation():
    return albumentations.Compose([albumentations.HorizontalFlip(p=0.5),
                                   albumentations.ShiftScaleRotate(scale_limit=0.5,
                                                                   rotate_limit=0,
                                                                   shift_limit=0.1,
                                                                   p=0.5,
                                                                   border_mode=0
                                                                   ),
                                   albumentations.GridDistortion(p=0.5),
                                   albumentations.Resize(320, 640),
                                   albumentations.Normalize(mean=(0.485, 0.456, 0.406),
                                                            std=(0.229, 0.224, 0.225)),
                                   ])

  @staticmethod
  def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    return albumentations.Compose([albumentations.Resize(320, 640),
                                   albumentations.Normalize(mean=(0.485, 0.456, 0.406),
                                                            std=(0.229, 0.224, 0.225)),
                                   ])

  def __init__(self,
               df_path: Path,
               resized_loc,
               subset: str = "train",
               N_FOLDS=10,
               SEED=246232,
               ):

    dataset_csv = pandas.read_csv(df_path)
    dataset_csv["label"] = dataset_csv["Image_Label"].apply(lambda x:x.split("_")[1])
    dataset_csv["im_id"] = dataset_csv["Image_Label"].apply(lambda x:x.split("_")[0])
    self.dataset_df = dataset_csv

    if subset != 'test':

      id_mask_count = (dataset_csv.loc[dataset_csv["EncodedPixels"].isnull() == False, "Image_Label"]
                       .apply(lambda x:x.split("_")[0])
                       .value_counts()
                       .sort_index()
                       .reset_index()
                       .rename(columns={"index":"img_id", "Image_Label":"count"})
                       )  # split data into train and val

      ids = id_mask_count["img_id"].values
      li = [[train_index, test_index]
            for train_index, test_index
            in StratifiedKFold(n_splits=N_FOLDS,
                               random_state=SEED
                               ).split(ids, id_mask_count["count"])
            ]

      self.data_folder = resized_loc / 'train_images_525'

      if subset == 'valid':
        self.img_ids = ids[li[0][1]]
      else:
        self.img_ids = ids[li[0][0]]
    else:
      self.img_ids = dataset_csv["Image_Label"].apply(lambda x:x.split("_")[0]).drop_duplicates().values
      self.data_folder = resized_loc / 'test_images_525'

    if subset == 'train':
      self.transforms = self.get_training_augmentation()
    else:
      self.transforms = self.get_validation_augmentation()

  classes = {0:"Fish", 1:"Flower", 2:"Gravel", 3:"Sugar"}
  predictors_shape = (350, 525, 3)
  response_shape = (len(classes),)

  def make_mask(self,
                df: pandas.DataFrame,
                resized_loc,
                image_name: str = "img.jpg"):
    """
    Create mask based on df, image name and shape.
    """
    masks = numpy.zeros((self.predictors_shape[0], self.predictors_shape[1], self.response_shape[0]),
                        dtype=numpy.float32)
    df = df[df["im_id"] == image_name]
    for idx, im_name in enumerate(df["im_id"].values):
      for classidx, classid in enumerate(self.classes.values()):
        mask = cv2.imread(str(resized_loc / 'train_masks_525' / 'train_masks_525' / f'{classid}{im_name}'))
        if mask is None:
          continue
        if mask[:, :, 0].shape != (self.predictors_shape[0], self.predictors_shape[1]):
          mask = cv2.resize(mask, (self.predictors_shape[1], self.predictors_shape[9]))
        masks[:, :, classidx] = mask[:, :, 0]
    masks = masks / 255
    return masks

  def __getitem__(self, idx):
    image_name = self.img_ids[idx]
    mask = self.make_mask(self.dataset_df, image_name)
    img = cv2.imread(str(self.data_folder / image_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    augmented = self.transforms(image=img, mask=mask)
    img = numpy.transpose(augmented["image"], [2, 0, 1])
    mask = numpy.transpose(augmented["mask"], [2, 0, 1])
    return img, mask

  def __len__(self):
    return len(self.img_ids)


def resize_it(x):
  if x.shape != (350, 525):
    x = cv2.resize(x, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
  return x


def make_mask_T(df, image_label, shape=(1400, 2100), cv_shape=(525, 350)):
  """
  Create mask based on df, image name and shape.
  """
  df = df.set_index('Image_Label')
  encoded_mask = df.loc[image_label, 'EncodedPixels']
  mask = numpy.zeros((shape[0], shape[1]), dtype=numpy.float32)
  if encoded_mask is not numpy.nan:
    mask = run_length_encoding2mask(encoded_mask, shape=shape)  # original size
  return cv2.resize(mask, cv_shape)


def post_process_minsize(mask, min_size):
  """
  Post processing of each predicted mask, components with lesser number of pixels
  than `min_size` are ignored
  """
  num_component, component = cv2.connectedComponents(mask.astype(numpy.uint8))
  predictions, num = numpy.zeros(mask.shape), 0
  for c in range(1, num_component):
    p = (component == c)
    if p.sum() > min_size:
      predictions[p] = 1
      num += 1
  return predictions


def visualize(image,
              mask,
              original_image=None,
              original_mask=None):
  """
  Plot image and masks.
  If two pairs of images and masks are passes, show both.
  """
  fontsize = 14

  if original_image is None and original_mask is None:
    f, ax = pyplot.subplots(1, 5, figsize=(24, 24))

    ax[0].imshow(image)
    for i in range(4):
      ax[i + 1].imshow(mask[:, :, i])
      ax[i + 1].set_title(f"Mask {CloudDataset.classes[i]}", fontsize=fontsize)
  else:
    f, ax = pyplot.subplots(2, 5, figsize=(24, 12))

    ax[0, 0].imshow(original_image)
    ax[0, 0].set_title("Original image", fontsize=fontsize)

    for i in range(4):
      ax[0, i + 1].imshow(original_mask[:, :, i])
      ax[0, i + 1].set_title(f"Original mask {CloudDataset.classes[i]}", fontsize=fontsize)

    ax[1, 0].imshow(image)
    ax[1, 0].set_title("Transformed image", fontsize=fontsize)

    for i in range(4):
      ax[1, i + 1].imshow(mask[:, :, i])
      ax[1, i + 1].set_title(f"Transformed mask {CloudDataset.classes[i]}", fontsize=fontsize)


def visualize_with_raw(image,
                       mask,
                       original_image=None,
                       original_mask=None,
                       raw_image=None,
                       raw_mask=None
                       ):
  """
  Plot image and masks.
  If two pairs of images and masks are passes, show both.
  """
  fontsize = 14

  f, ax = pyplot.subplots(3, 5, figsize=(24, 12))

  ax[0, 0].imshow(original_image)
  ax[0, 0].set_title("Original image", fontsize=fontsize)

  for i in range(4):
    ax[0, i + 1].imshow(original_mask[:, :, i])
    ax[0, i + 1].set_title(f"Original mask {CloudDataset.classes[i]}", fontsize=fontsize)

  ax[1, 0].imshow(raw_image)
  ax[1, 0].set_title("Original image", fontsize=fontsize)

  for i in range(4):
    ax[1, i + 1].imshow(raw_mask[:, :, i])
    ax[1, i + 1].set_title(f"Raw predicted mask {CloudDataset.classes[i]}", fontsize=fontsize)

  ax[2, 0].imshow(image)
  ax[2, 0].set_title("Transformed image", fontsize=fontsize)

  for i in range(4):
    ax[2, i + 1].imshow(mask[:, :, i])
    ax[2, i + 1].set_title(
      f"Predicted mask with processing {CloudDataset.classes[i]}", fontsize=fontsize
      )


def plot_with_augmentation(image, mask, augment):
  """
  Wrapper for `visualize` function.
  """
  augmented = augment(image=image, mask=mask)
  image_flipped = augmented["image"]
  mask_flipped = augmented["mask"]
  visualize(image_flipped, mask_flipped, original_image=image, original_mask=mask)


def post_process(probability, threshold, min_size):
  """
  This is slightly different from other kernels as we draw convex hull here itself.
  Post processing of each predicted mask, components with lesser number of pixels
  than `min_size` are ignored
  """
  mask = (cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1])
  mask = draw_convex_hull(mask.astype(numpy.uint8))
  num_component, component = cv2.connectedComponents(mask.astype(numpy.uint8))
  predictions = numpy.zeros((350, 525), numpy.float32)
  num = 0
  for c in range(1, num_component):
    p = component == c
    if p.sum() > min_size:
      predictions[p] = 1
      num += 1
  return predictions, num

