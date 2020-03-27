# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

from pathlib import Path
from typing import Tuple, Union

import numpy
import torch
from PIL import Image
from matplotlib import pyplot
from torchvision.transforms import Compose, Resize, ToTensor

from draugr.opencv_utilities import cv2_resize
from draugr.torch_utilities import (
    to_tensor,
    global_torch_device,
    uint_hwc_to_chw_float_tensor,
    float_chw_to_hwc_uint_tensor,
    hwc_to_chw_tensor,
)
from neodroidvision.data.datasets.supervised.splitting import Split
from neodroidvision.data.datasets.supervised.supervised_dataset import SupervisedDataset

__all__ = ["PennFudanDataset"]

from neodroidvision.utilities import (
    TupleCompose,
    TupleRandomHorizontalFlip,
    TupleToTensor,
)


class PennFudanDataset(SupervisedDataset):
    predictor_channels = 3  # RGB input
    response_channels = 2  # our dataset has two classes only - background and person
    response_channels_binary = 1

    image_size = (256, 256)
    image_size_T = image_size[::-1]

    @property
    def response_shape(self) -> Tuple[int, ...]:
        return (*self.image_size_T, self.response_channels_binary)

    @property
    def predictor_shape(self) -> Tuple[int, ...]:
        return (*self.image_size_T, self.predictor_channels)

    @staticmethod
    def get_transforms(split: Split):
        transforms = [Resize(PennFudanDataset.image_size_T), ToTensor()]

        # if split == Split.Training:
        #  transforms.append(RandomHorizontalFlip(0.5))

        return Compose(transforms)

    @staticmethod
    def get_tuple_transforms(split: Split):
        transforms = [
            # Resize(PennFudanDataset.image_size_T),
            TupleToTensor()
        ]

        if split == Split.Training:
            transforms.append(TupleRandomHorizontalFlip(0.5))

        return TupleCompose(transforms)

    def __init__(self, root: Union[str, Path], split: Split = Split.Training):
        """

    :param root:
    :type root:
    :param split:
    :type split:
    """
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        self._root_data_path = root
        self._transforms = self.get_transforms(split)
        self._img_path = root / "PNGImages"
        self._ped_path = root / "PedMasks"
        self.imgs = list(
            sorted(self._img_path.iterdir())
        )  # load all image files, sorting them to
        self.masks = list(
            sorted(self._ped_path.iterdir())
        )  # ensure that they are aligned

    def __getitem__(self, idx: int):
        """

    :param idx:
    :type idx:
    :return:
    :rtype:
    """
        # return self.get_instanced(idx)
        return self.get_binary(idx)

    def get_binary(self, idx):
        img = numpy.array(Image.open(self._img_path / self.imgs[idx]).convert("RGB"))
        mask = numpy.array(Image.open(self._ped_path / self.masks[idx]))

        mask[mask != 0] = 1.0

        img = cv2_resize(img, self.image_size_T)
        mask = cv2_resize(mask, self.image_size_T)

        return (
            uint_hwc_to_chw_float_tensor(to_tensor(img, dtype=torch.uint8)),
            to_tensor(mask).unsqueeze(0),
        )

    def get_instanced(self, idx):
        img = to_tensor(Image.open(self._img_path / self.imgs[idx]).convert("RGB"))
        mask = to_tensor(Image.open(self._ped_path / self.masks[idx]))

        obj_ids = torch.unique(mask)  # instances are encoded as different colors
        obj_ids = obj_ids[1:]  # first id is the background, so remove it

        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        return img, masks

    def get_all(self, idx):
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
    dataset = PennFudanDataset(
        Path.home() / "Data" / "Datasets" / "PennFudanPed", Split.Training
    )

    global_torch_device(override=global_torch_device(cuda_if_available=False))

    idx = -2
    img, mask = dataset[idx]
    print(img)
    print(img.shape, mask.shape)
    pyplot.imshow(float_chw_to_hwc_uint_tensor(img))
    pyplot.show()
    pyplot.imshow(mask.squeeze(0))
    pyplot.show()
