#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/03/2020
           """

from enum import Enum
from pathlib import Path
from typing import Tuple, Union

import numpy
import torch
from PIL import Image
from draugr.numpy_utilities import SplitEnum, chw_to_hwc
from draugr.numpy_utilities.mixing import mix_channels
from draugr.opencv_utilities import cv2_resize, InterpolationEnum, draw_masks
from draugr.opencv_utilities.bounding_boxes import draw_boxes
from draugr.torch_utilities import (
    SupervisedDataset,
    float_chw_to_hwc_uint_tensor,
    global_torch_device,
    to_tensor,
    uint_hwc_to_chw_float_tensor,
)
from matplotlib import pyplot
from sorcery import assigned_names
from torchvision.transforms import Compose, Resize, ToTensor

__all__ = ["PennFudanDataset"]

from neodroidvision.utilities import (
    TupleCompose,
    TupleRandomHorizontalFlip,
    TupleToTensor,
)


class PennFudanDataset(SupervisedDataset):
    """ """

    predictor_channels = 3  # RGB input
    response_channels_two_classes = (
        2  # our dataset has two classes only - background and person
    )
    response_channels_binary = 1
    response_channels_instanced = None

    image_size = (256, 256)
    image_size_T = image_size[::-1]

    categories = ("void", "person")

    class PennFudanReturnVariantEnum(Enum):
        """
        Return binary mask, instanced or all annotations
        """

        binary, instanced, all = assigned_names()

    @property
    def response_shape(self) -> Tuple[int, ...]:
        """

        :return:
        :rtype:"""
        if self._return_variant == PennFudanDataset.PennFudanReturnVariantEnum.binary:
            return (*self.image_size_T, self.response_channels_binary)
        elif (
            self._return_variant
            == PennFudanDataset.PennFudanReturnVariantEnum.instanced
        ):
            return (*self.image_size_T, self.response_channels_instanced)
        elif self._return_variant == PennFudanDataset.PennFudanReturnVariantEnum.all:
            return (*self.image_size_T, self.response_channels_two_classes)
        raise NotImplementedError

    @property
    def predictor_shape(self) -> Tuple[int, ...]:
        """

        :return:
        :rtype:"""
        return (*self.image_size_T, self.predictor_channels)

    @staticmethod
    def get_transforms(split: SplitEnum):
        """

        :param split:
        :type split:
        :return:
        :rtype:"""
        transforms = [Resize(PennFudanDataset.image_size_T), ToTensor()]

        # if split == SplitEnum.training:
        #  transforms.append(RandomHorizontalFlip(0.5))

        return Compose(transforms)

    @staticmethod
    def get_tuple_transforms(split: SplitEnum):
        """

        :param split:
        :type split:
        :return:
        :rtype:"""
        transforms = [
            # Resize(PennFudanDataset.image_size_T),
            TupleToTensor()
        ]

        if split == SplitEnum.training:
            transforms.append(TupleRandomHorizontalFlip(0.5))

        return TupleCompose(transforms)

    def __init__(
        self,
        root: Union[str, Path],
        split: SplitEnum = SplitEnum.training,
        return_variant: PennFudanReturnVariantEnum = PennFudanReturnVariantEnum.binary,
    ):
        """

        :param root:
        :type root:
        :param split:
        :type split:"""
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        self._root_data_path = root
        self._return_variant = return_variant

        if self._return_variant != PennFudanDataset.PennFudanReturnVariantEnum.all:
            self._transforms = self.get_transforms(split)
        else:
            self._transforms = self.get_tuple_transforms(split)

        if self._return_variant == PennFudanDataset.PennFudanReturnVariantEnum.binary:
            self._getter = self.get_binary
        elif (
            self._return_variant
            == PennFudanDataset.PennFudanReturnVariantEnum.instanced
        ):
            self._getter = self.get_instanced
        elif self._return_variant == PennFudanDataset.PennFudanReturnVariantEnum.all:
            self._getter = self.get_all
        else:
            raise NotImplementedError

        self._img_path = root / "PNGImages"
        self._ped_path = root / "PedMasks"
        self.imgs = list(
            sorted(self._img_path.iterdir())
        )  # load all image files, sorting them to
        self.masks = list(
            sorted(self._ped_path.iterdir())
        )  # ensure that they are aligned
        if (
            self._return_variant
            == PennFudanDataset.PennFudanReturnVariantEnum.instanced
        ):
            max_num_instance = 0
            for m in self.masks:
                mask = numpy.array(Image.open(self._ped_path / m))
                num_unique = numpy.unique(mask).shape[0]
                if max_num_instance < num_unique:
                    max_num_instance = num_unique
            PennFudanDataset.response_channels_instanced = max_num_instance
            self.zero_mask = numpy.zeros(
                self.response_shape[::-1]
            )  # reversed order numpy array of torch tensor output

    def __getitem__(self, idx: int):
        """

        :param idx:
        :type idx:
        :return:
        :rtype:"""
        return self._getter(idx)

    def get_binary(self, idx):
        """
        Return a single binary channel target for all instances in image

        :param idx:
        :type idx:
        :return:
        :rtype:"""
        img = numpy.array(Image.open(self._img_path / self.imgs[idx]).convert("RGB"))
        mask = numpy.array(Image.open(self._ped_path / self.masks[idx]))

        mask[mask != 0] = 1.0

        img = cv2_resize(img, self.image_size_T)
        mask = cv2_resize(mask, self.image_size_T, InterpolationEnum.nearest)

        return (
            uint_hwc_to_chw_float_tensor(to_tensor(img, dtype=torch.uint8)),
            to_tensor(mask).unsqueeze(0),
        )

    def get_instanced(self, idx):
        """
        Return a separate channel target for each instance in image

        :param idx:
        :type idx:
        :return:
        :rtype:"""
        img = numpy.array(Image.open(self._img_path / self.imgs[idx]).convert("RGB"))
        mask = numpy.array(Image.open(self._ped_path / self.masks[idx]))

        img = cv2_resize(img, self.image_size_T)
        mask = cv2_resize(mask, self.image_size_T, InterpolationEnum.nearest)

        obj_ids = numpy.unique(mask)  # instances are encoded as different colors
        obj_ids = obj_ids[1:]  # first id is the background, so remove it

        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]
        zero_mask_clone = self.zero_mask.copy()
        zero_mask_clone[: masks.shape[0]] = masks

        return (
            uint_hwc_to_chw_float_tensor(to_tensor(img, dtype=torch.uint8)),
            torch.as_tensor(zero_mask_clone, dtype=torch.uint8),
        )

    def get_all(self, idx):
        """
        Return all info including bounding boxes for each instance

        :param idx:
        :type idx:
        :return:
        :rtype:"""
        mask = torch.as_tensor(
            numpy.array(Image.open(self._ped_path / self.masks[idx]))
        )
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background

        obj_ids = torch.unique(mask)  # instances are encoded as different colors
        obj_ids = obj_ids[1:]  # first id is the background, so remove it

        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = torch.where(masks[i])
            xmin = torch.min(pos[1])
            xmax = torch.max(pos[1])
            ymin = torch.min(pos[0])
            ymax = torch.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)  # there is only one class
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # TODO: IMPLEMENT RESIZING OF PICTURES

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        is_crowd = torch.zeros(
            (num_objs,), dtype=torch.int64
        )  # suppose all instances are not crowd

        return self._transforms(
            Image.open(self._img_path / self.imgs[idx]).convert("RGB"),
            dict(
                boxes=boxes,
                labels=labels,
                masks=masks,
                image_id=image_id,
                area=area,
                iscrowd=is_crowd,
            ),
        )

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":

    def main_binary(p=Path.home() / "Data" / "Datasets" / "PennFudanPed"):

        dataset = PennFudanDataset(p, SplitEnum.training)

        global_torch_device(override=global_torch_device("cpu"))

        idx = -2
        img, mask = dataset[idx]
        print(img)
        print(img.shape, mask.shape)
        pyplot.imshow(float_chw_to_hwc_uint_tensor(img))
        pyplot.show()
        pyplot.imshow(mask.squeeze(0))
        pyplot.show()

    def main_instanced(p=Path.home() / "Data" / "Datasets" / "PennFudanPed"):

        dataset = PennFudanDataset(
            p,
            SplitEnum.training,
            return_variant=PennFudanDataset.PennFudanReturnVariantEnum.instanced,
        )

        global_torch_device(override=global_torch_device("cpu"))

        idx = -2
        img, mask = dataset[idx]
        print(img)
        print(img.shape, mask.shape)
        pyplot.imshow(float_chw_to_hwc_uint_tensor(img))
        pyplot.show()
        for m in mask:
            pyplot.imshow(m.squeeze(0))
            pyplot.show()

    def main_instanced_mixed(p=Path.home() / "Data" / "Datasets" / "PennFudanPed"):

        dataset = PennFudanDataset(
            p,
            SplitEnum.training,
            return_variant=PennFudanDataset.PennFudanReturnVariantEnum.instanced,
        )

        global_torch_device(override=global_torch_device("cpu"))

        idx = -2
        img, mask = dataset[idx]
        print(img)
        print(img.shape, mask.shape)
        pyplot.imshow(float_chw_to_hwc_uint_tensor(img))
        pyplot.show()
        print(mask.shape)
        pyplot.imshow(mix_channels(chw_to_hwc(mask.numpy())))
        pyplot.show()

    def main_instanced_single_channel(
        p=Path.home() / "Data" / "Datasets" / "PennFudanPed",
    ):

        dataset = PennFudanDataset(
            p,
            SplitEnum.training,
            return_variant=PennFudanDataset.PennFudanReturnVariantEnum.instanced,
        )

        global_torch_device(override=global_torch_device("cpu"))

        idx = -2
        img, mask = dataset[idx]
        print(img)
        print(img.shape, mask.shape)
        i = float_chw_to_hwc_uint_tensor(img).numpy()
        # pyplot.imshow(i)
        # pyplot.show()
        a, b = numpy.zeros_like(mask), mask.numpy()
        print(a.shape, b.shape)
        pyplot.imshow(draw_masks(i, b))
        pyplot.show()

    def main_all_bb(p=Path.home() / "Data" / "Datasets" / "PennFudanPed"):

        dataset = PennFudanDataset(
            p,
            SplitEnum.training,
            return_variant=PennFudanDataset.PennFudanReturnVariantEnum.all,
        )

        global_torch_device(override=global_torch_device("cpu"))

        idx = -2
        img, info = dataset[idx]
        print(img)
        print(img.shape)

        img = float_chw_to_hwc_uint_tensor(img).detach().numpy()
        pyplot.imshow(
            draw_boxes.draw_bounding_boxes(
                img, info["boxes"], labels=info["labels"], mode="RGB"
            )
        )
        pyplot.show()

    # p =         Path.home() / "Data3" / "PennFudanPed"
    p = Path.home() / "Data" / "Datasets" / "PennFudanPed"
    # main_binary(p)
    # main_instanced(p)
    # main_instanced_mixed(p)
    main_instanced_single_channel(p)
    # main_all_bb(p    )
