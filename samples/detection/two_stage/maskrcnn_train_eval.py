#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path

import torch
from PIL import Image

__author__ = "Christian Heider Nielsen"
__doc__ = ""

from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from draugr.torch_utilities import (
    global_torch_device,
    torch_seed,
    TorchEvalSession,
    TorchTrainSession,
)
from draugr.python_utilities.functions import collate_batch_fn
from neodroidvision.data.datasets.supervised.segmentation import PennFudanDataset
from neodroidvision.data.datasets.supervised.segmentation.penn_fudan import (
    ReturnVariant,
)
from draugr.torch_utilities import Split, SplitByPercentage
from neodroidvision.detection.two_stage.mask_rcnn.architecture import (
    get_pretrained_instance_segmentation_maskrcnn,
)
from neodroidvision.detection.two_stage.mask_rcnn.maskrcnn_engine import (
    train_single_epoch,
    maskrcnn_evaluate,
)
from warg import GDKC

if __name__ == "__main__":
    dataset_root = Path("/home/heider/Data/Datasets")

    batch_size = 2
    test_split_count = 50
    num_epochs = 10
    optimiser_spec = GDKC(torch.optim.SGD, lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler_spec = GDKC(
        torch.optim.lr_scheduler.StepLR,
        # a learning rate scheduler which decreases the learning rate by
        step_size=3,  # 10x every 3 epochs
        gamma=0.1,
    )
    num_workers = os.cpu_count()
    torch_seed(3825)

    dataset = PennFudanDataset(
        dataset_root / "PennFudanPed", Split.Training, return_variant=ReturnVariant.all
    )
    dataset_test = PennFudanDataset(
        dataset_root / "PennFudanPed", Split.Testing, return_variant=ReturnVariant.all
    )
    split = SplitByPercentage(len(dataset))

    split_indices = torch.randperm(split.total_num).tolist()

    data_loader = DataLoader(
        Subset(dataset, split_indices[: -split.testing_num]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_batch_fn,
    )

    data_loader_test = DataLoader(
        Subset(dataset_test, split_indices[-split.testing_num]),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch_fn,
    )

    model = get_pretrained_instance_segmentation_maskrcnn(dataset.response_channels)
    optimizer = optimiser_spec([p for p in model.parameters() if p.requires_grad])
    lr_scheduler = scheduler_spec(optimizer)

    with TorchTrainSession(model):
        for epoch_i in tqdm(range(num_epochs), desc="Epoch #"):
            # train for one epoch, printing every 10 iterations
            train_single_epoch(model, optimizer, data_loader, epoch_i, log_frequency=10)
            lr_scheduler.step()  # update the learning rate
            maskrcnn_evaluate(model, data_loader_test)  # evaluate on the test dataset

    with TorchEvalSession(model):  # put the model in evaluation mode
        img, _ = dataset_test[0]  # pick one image from the test set

        with torch.no_grad():
            prediction = model([img.to(global_torch_device())])

        Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

        Image.fromarray(prediction[0]["masks"][0, 0].mul(255).byte().cpu().numpy())
