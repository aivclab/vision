#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 27/06/2020
           """

import csv
from pathlib import Path
from typing import Dict, Tuple

import torch
from PIL import Image
from draugr.numpy_utilities import SplitEnum
from matplotlib import pyplot
from torch.utils import data
from torchvision import transforms

__all__ = ["VggFace2"]

from draugr.torch_utilities import SupervisedDataset


class VggFace2(SupervisedDataset):
    """
    Department of Engineering Science, University of Oxford
    Visual Geometry Group Face 2 Dataset
    """

    """"""

    @property
    def response_shape(self) -> Tuple[int, ...]:
        """

        :return:
        :rtype:"""
        return (0,)

    @property
    def predictor_shape(self) -> Tuple[int, ...]:
        """

        :return:
        :rtype:"""
        return self._resize_shape

    # mean = numpy.array([0.485, 0.456, 0.406])
    # std = numpy.array([0.229, 0.224, 0.225])

    inverse_transform = transforms.Compose(
        [
            # transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist()),
            transforms.ToPILImage()
        ]
    )

    @staticmethod
    def get_id_label_map(meta_file: Path):
        """

        :param meta_file:
        :type meta_file:
        :return:
        :rtype:"""
        import pandas

        N_IDENTITY = 9131  # total number of identities in VGG Face2
        N_IDENTITY_PRETRAIN = 8631  # the number of identities used in training by Caffe
        identity_list = meta_file
        df = pandas.read_csv(
            identity_list,
            sep=",\s+",
            quoting=csv.QUOTE_ALL,
            encoding="utf-8",
            engine="python",
        )
        df["class"] = -1
        df.loc[df["Flag"] == 1, "class"] = range(N_IDENTITY_PRETRAIN)
        df.loc[df["Flag"] == 0, "class"] = range(N_IDENTITY_PRETRAIN, N_IDENTITY)

        key = df["Class_ID"].values
        val = df["class"].values
        id_label_dict = dict(zip(key, val))
        return id_label_dict

    @property
    def split_names(self) -> Dict[SplitEnum, str]:
        """

        :return:
        :rtype:"""
        return {
            SplitEnum.training: "train",
            SplitEnum.validation: "validation",
            SplitEnum.testing: "test",
        }

    def __init__(
        self,
        dataset_path: Path,
        split: SplitEnum = SplitEnum.training,
        *,
        resize_s: int = 256,
        raw_images: bool = False,
        verbose: bool = False,
    ):
        """
        :type resize_s: int or tuple(w,h)
        :param dataset_path: dataset directory
        :param split: train, valid, test"""
        super().__init__()
        assert dataset_path.exists(), f"root: {dataset_path} not found."
        split = self.split_names[split]
        if isinstance(resize_s, int):
            assert resize_s > 2, "resize_s should be >2"
            resize_s = (resize_s, resize_s, 3)
        self._resize_shape = (*resize_s, 3)

        self._dataset_path = dataset_path / split
        image_list_file_path = dataset_path / f"{split}_list.txt"
        assert (
            image_list_file_path.exists()
        ), f"image_list_file: {image_list_file_path} not found."

        self._image_list_file_path = image_list_file_path
        meta_id_path = dataset_path / "identity_meta.csv"
        if not meta_id_path.exists():
            meta_id_path = dataset_path.parent / "meta" / meta_id_path.name
        assert meta_id_path.exists(), f"meta id path {meta_id_path} does not exists"

        self._split = split
        self._id_label_dict = self.get_id_label_map(meta_id_path)
        self._return_raw_images = raw_images

        self.train_trans = transforms.Compose(
            [
                transforms.RandomResizedCrop(self._resize_shape[:2]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize(self.mean, self.std)
            ]
        )

        self.val_trans = transforms.Compose(
            [
                transforms.Resize(self._resize_shape[:2]),
                transforms.CenterCrop(self._resize_shape[:2]),
                transforms.ToTensor(),
                # transforms.Normalize(self.mean, self.std)
            ]
        )

        self._img_info = []
        with open(str(self._image_list_file_path), "r") as f:
            for i, img_file in enumerate(f):
                img_file = img_file.strip()  # e.g. n004332/0317_01.jpg
                class_id = img_file.split("/")[0]  # like n004332
                label = self._id_label_dict[class_id]
                self._img_info.append(
                    {"class_id": class_id, "img": img_file, "label": label}
                )
                if verbose and i % 1000 == 0:
                    print(f"Processing: {i} images for {self._split} split")

    def __len__(self):
        return len(self._img_info)

    def __getitem__(self, index):
        info = self._img_info[index]
        img_file = info["img"]
        img = Image.open(str(self._dataset_path / img_file))

        if not self._return_raw_images:
            if self._split == SplitEnum.training:
                img = self.train_trans(img)
            else:
                img = self.val_trans(img)

        label = info["label"]
        class_id = info["class_id"]

        return img, label, img_file, class_id


if __name__ == "__main__":

    def main():
        import tqdm

        batch_size = 32

        dt = VggFace2(
            Path.home() / "Data" / "VGG-Face2" / "data",
            split=SplitEnum.testing,
            # raw_images=True
        )

        test_loader = torch.utils.data.DataLoader(
            dt, batch_size=batch_size, shuffle=False
        )

        # test_loader = dt

        for batch_idx, (imgs, label, img_files, class_ids) in tqdm.tqdm(
            enumerate(test_loader),
            total=len(test_loader),
            desc=f"{test_loader.dataset}",
            ncols=80,
            leave=False,
        ):
            pyplot.imshow(dt.inverse_transform(imgs[0]))
            pyplot.title(f"{label[0], class_ids[0]}")
            # pyplot.imshow(imgs)
            pyplot.show()
            break

    main()
