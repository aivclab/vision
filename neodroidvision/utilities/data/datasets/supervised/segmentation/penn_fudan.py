# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

from pathlib import Path
from typing import Union

import numpy
import torch
import torchvision
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from neodroidvision.utilities.data.datasets.supervised.detection.coco.coco_utilities import (
    Compose,
    RandomHorizontalFlip,
    ToTensor,
)
from neodroidvision.utilities.data.datasets.supervised.supervised_dataset import (
    Split,
    SupervisedDataset,
)

__all__ = ["PennFudanDataset"]


class PennFudanDataset(SupervisedDataset):
    num_categories = 2  # our dataset has two classes only - background and person

    @staticmethod
    def get_transform(split: Split):
        transforms = []
        transforms.append(ToTensor())
        if split == Split.Training:
            transforms.append(RandomHorizontalFlip(0.5))
        return Compose(transforms)

    def __init__(self, root: Union[str, Path], split: Split = Split.Training):
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        self.root = root
        self.transforms = self.get_transform(split)
        # load all image files, sorting them to
        # ensure that they are aligned
        self._img_path = root / "PNGImages"
        self._ped_path = root / "PedMasks"
        self.imgs = list(sorted(self._img_path.iterdir()))
        self.masks = list(sorted(self._ped_path.iterdir()))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self._img_path / self.imgs[idx]
        mask_path = self._ped_path / self.masks[idx]
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = numpy.array(mask)
        # instances are encoded as different colors
        obj_ids = numpy.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = numpy.where(masks[i])
            xmin = numpy.min(pos[1])
            xmax = numpy.max(pos[1])
            ymin = numpy.min(pos[0])
            ymax = numpy.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return self.transforms(img, target)

    def __len__(self):
        return len(self.imgs)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


if __name__ == "__main__":
    dataset = PennFudanDataset("/home/heider/Data/Datasets/PennFudanPed", False)

    print(dataset[-1])
