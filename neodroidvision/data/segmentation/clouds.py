#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 30/11/2019
           """

from pathlib import Path

import albumentations
import cv2
import numpy
import pandas
from matplotlib import pyplot
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

__all__ = ["CloudSegmentationDataset"]

from draugr.numpy_utilities import (
    uint_hwc_to_chw_float,
    hwc_to_chw,
    float_chw_to_hwc_uint,
    chw_to_hwc,
    SplitEnum,
)
from draugr.opencv_utilities import cv2_resize


class CloudSegmentationDataset(Dataset):
    """ """

    categories = {0: "Fish", 1: "Flower", 2: "Gravel", 3: "Sugar"}
    image_size = (640, 320)
    image_size_T = image_size[::-1]

    predictor_channels = 3
    response_channels = len(categories)

    predictors_shape = (*image_size_T, predictor_channels)
    response_shape = (*image_size_T, response_channels)

    predictors_shape_T = predictors_shape[::-1]
    response_shape_T = response_shape[::-1]

    mean = (0.2606705, 0.27866408, 0.32657165)  # Computed prior
    std = (0.25366131, 0.24921637, 0.23504028)  # Computed prior

    def training_augmentations(self):
        """

        Returns:

        """
        return [
            albumentations.VerticalFlip(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.ShiftScaleRotate(p=0.5, rotate_limit=0, border_mode=0),
        ]

    def validation_augmentations(self):
        """Add paddings to make image shape divisible by 32"""
        return [
            albumentations.Resize(*self.image_size_T),
            # albumentations.Normalize(mean=self.mean, std=self.std)
            # Standardization
        ]

    '''
def un_standardise(self, img):
"""Add paddings to make image shape divisible by 32"""
return (img * self.std + self.mean).astype(numpy.uint8)
'''

    def __init__(
        self,
        csv_path: Path,
        image_data_path: Path,
        subset: SplitEnum = SplitEnum.training,
        transp=True,
        N_FOLDS=10,
        SEED=246232,
    ):

        self.transp = transp

        if subset != subset.testing:
            data_frame = pandas.read_csv(csv_path / f"train.csv")
        else:
            data_frame = pandas.read_csv(csv_path / f"sample_submission.csv")

        data_frame["label"] = data_frame["Image_Label"].apply(lambda x: x.split("_")[1])
        data_frame["im_id"] = data_frame["Image_Label"].apply(lambda x: x.split("_")[0])
        self.data_frame = data_frame
        self.subset = subset
        self.base_image_data = image_data_path

        if subset != subset.testing:
            id_mask_count = (
                data_frame.loc[
                    data_frame["EncodedPixels"].isnull() == False, "Image_Label"
                ]
                .apply(lambda x: x.split("_")[0])
                .value_counts()
                .sort_index()
                .reset_index()
                .rename(columns={"index": "img_id", "Image_Label": "count"})
            )  # split data into train and val

            ids = id_mask_count["img_id"].values
            li = [
                [train_index, test_index]
                for train_index, test_index in StratifiedKFold(
                    n_splits=N_FOLDS, random_state=SEED
                ).split(ids, id_mask_count["count"])
            ]

            self.image_data_path = image_data_path / "train_images_525"

            if subset == SplitEnum.validation:
                self.img_ids = ids[li[0][1]]
            else:
                self.img_ids = ids[li[0][0]]
        else:
            self.img_ids = (
                data_frame["Image_Label"]
                .apply(lambda x: x.split("_")[0])
                .drop_duplicates()
                .values
            )
            self.image_data_path = image_data_path / "test_images_525"

        if subset == SplitEnum.training:
            self.transforms = albumentations.Compose(
                self.training_augmentations()  # +
                #                   self.validation_augmentations()
            )
        else:
            self.transforms = albumentations.Compose(self.validation_augmentations())

    def fetch_masks(self, image_name: str):
        """
        Create mask based on df, image name and shape."""
        masks = numpy.zeros(self.response_shape, dtype=numpy.float32)
        df = self.data_frame[self.data_frame["im_id"] == image_name]

        for idx, im_name in enumerate(df["im_id"].values):
            for classidx, classid in enumerate(self.categories.values()):
                mpath = str(
                    self.base_image_data / "train_masks_525" / f"{classid}{im_name}"
                )
                mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                mask = cv2_resize(mask, self.image_size_T)
                masks[..., classidx] = mask

        return masks / 255.0

    @staticmethod
    def no_info_mask(img):
        """

        Args:
          img:

        Returns:

        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = numpy.array([0, 0, 0], numpy.uint8)
        upper = numpy.array([180, 255, 10], numpy.uint8)
        return (~(cv2.inRange(hsv, lower, upper) > 250)).astype(numpy.uint8)

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        masks = self.fetch_masks(image_name)
        img = cv2.imread(str(self.image_data_path / image_name))
        img = cv2_resize(img, self.image_size_T)
        img_o = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            augmented = self.transforms(image=img_o, mask=masks)
            img_o = augmented["image"]
            masks = augmented["mask"]
        img_o = uint_hwc_to_chw_float(img_o)
        masks = hwc_to_chw(masks)
        if self.subset == SplitEnum.testing:
            return img_o, masks, self.no_info_mask(img)
        return img_o, masks

    def __len__(self):
        return len(self.img_ids)

    @staticmethod
    def visualise(image, mask, original_image=None, original_mask=None):
        """
        Plot image and masks.
        If two pairs of images and masks are passes, show both."""
        fontsize = 14

        if original_image is None and original_mask is None:
            f, ax = pyplot.subplots(1, 5, figsize=(24, 24))

            ax[0].imshow(image)
            for i in range(4):
                ax[i + 1].imshow(mask[..., i])
                ax[i + 1].set_title(
                    f"Mask {CloudSegmentationDataset.categories[i]}", fontsize=fontsize
                )
        else:
            f, ax = pyplot.subplots(2, 5, figsize=(24, 12))

            ax[0, 0].imshow(original_image)
            ax[0, 0].set_title("Original image", fontsize=fontsize)

            for i in range(4):
                ax[0, i + 1].imshow(original_mask[..., i], vmin=0, vmax=1)
                ax[0, i + 1].set_title(
                    f"Original mask {CloudSegmentationDataset.categories[i]}",
                    fontsize=fontsize,
                )

            ax[1, 0].imshow(image)
            ax[1, 0].set_title("Transformed image", fontsize=fontsize)

            for i in range(4):
                ax[1, i + 1].imshow(mask[..., i], vmin=0, vmax=1)
                ax[1, i + 1].set_title(
                    f"Transformed mask {CloudSegmentationDataset.categories[i]}",
                    fontsize=fontsize,
                )

        pyplot.show()

    @staticmethod
    def visualise_prediction(
        processed_image,
        processed_mask,
        original_image=None,
        original_mask=None,
        raw_image=None,
        raw_mask=None,
    ):
        """
        Plot image and masks.
        If two pairs of images and masks are passes, show both."""
        fontsize = 14

        f, ax = pyplot.subplots(3, 5, figsize=(24, 12))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title("Original image", fontsize=fontsize)

        for i in range(4):
            ax[0, i + 1].imshow(original_mask[..., i], vmin=0, vmax=1)
            ax[0, i + 1].set_title(
                f"Original mask {CloudSegmentationDataset.categories[i]}",
                fontsize=fontsize,
            )

        ax[1, 0].imshow(raw_image)
        ax[1, 0].set_title("Raw image", fontsize=fontsize)

        for i in range(4):
            ax[1, i + 1].imshow(raw_mask[..., i], vmin=0, vmax=1)
            ax[1, i + 1].set_title(
                f"Predicted mask {CloudSegmentationDataset.categories[i]}",
                fontsize=fontsize,
            )

        ax[2, 0].imshow(processed_image)
        ax[2, 0].set_title("Transformed image", fontsize=fontsize)

        for i in range(4):
            ax[2, i + 1].imshow(processed_mask[..., i])
            ax[2, i + 1].set_title(
                f"Predicted mask with processing {CloudSegmentationDataset.categories[i]}",
                fontsize=fontsize,
            )

        pyplot.show()

    def plot_training_sample(self):
        """
        Wrapper for `visualize` function."""
        orig_transforms = self.transforms
        self.transforms = None
        image, mask = self.__getitem__(numpy.random.randint(0, self.__len__()))
        print(image.shape)
        print(mask.shape)
        self.transforms = orig_transforms
        image = float_chw_to_hwc_uint(image)
        mask = chw_to_hwc(mask)
        print(image.shape)
        print(mask.shape)
        augmented = orig_transforms(image=image, mask=mask)
        augmented_image = augmented["image"]
        augmented_mask = augmented["mask"]
        print(augmented_image.shape)
        print(augmented_mask.shape)
        self.visualise(
            augmented_image, augmented_mask, original_image=image, original_mask=mask
        )


if __name__ == "__main__":
    base_path = Path.home() / "Data" / "Datasets" / "Clouds"
    resized_loc = base_path / "resized"

    ds = CloudSegmentationDataset(base_path, resized_loc)
    ds.plot_training_sample()
