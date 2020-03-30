#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import numpy
import torch
from matplotlib import pyplot
from torch.utils.data import DataLoader
from tqdm import tqdm

from draugr.opencv_utilities import cv2_resize
from draugr.torch_utilities import (
    TorchCacheSession,
    TorchEvalSession,
    TorchTrainSession,
    global_torch_device,
    torch_seed,
)
from neodroidvision import PROJECT_APP_PATH
from neodroidvision.multitask.fission.skip_hourglass import SkipHourglassFission
from neodroidvision.segmentation import BCEDiceLoss
from neodroidvision.segmentation.evaluation.iou import intersection_over_union

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """

from neodroidvision.data.datasets.supervised.segmentation import PennFudanDataset
from neodroidvision.data.datasets.supervised import Split


def reschedule(model, epoch, scheduler):
    "This can be improved its just a hacky way to write SGDWR "
    if epoch == 7:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
        current_lr = next(iter(optimizer.param_groups))["lr"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 6, eta_min=current_lr / 100, last_epoch=-1
        )
    if epoch == 13:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
        current_lr = next(iter(optimizer.param_groups))["lr"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 6, eta_min=current_lr / 100, last_epoch=-1
        )
    if epoch == 19:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.002)
        current_lr = next(iter(optimizer.param_groups))["lr"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 6, eta_min=current_lr / 100, last_epoch=-1
        )
    if epoch == 25:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.002)
        current_lr = next(iter(optimizer.param_groups))["lr"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 6, eta_min=current_lr / 100, last_epoch=-1
        )

    return model, scheduler


def train_segmenter(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    scheduler,
    save_model_path: Path,
    n_epochs: int = 100,
):
    """

:param model:
:type model:
:param train_loader:
:type train_loader:
:param valid_loader:
:type valid_loader:
:param criterion:
:type criterion:
:param optimizer:
:type optimizer:
:param scheduler:
:type scheduler:
:param save_model_path:
:type save_model_path:
:param n_epochs:
:type n_epochs:
:return:
:rtype:
"""
    valid_loss_min = numpy.Inf  # track change in validation loss
    assert n_epochs > 0, n_epochs
    E = tqdm(range(1, n_epochs + 1))
    for epoch in E:
        train_loss = 0.0
        valid_loss = 0.0
        dice_score = 0.0

        for data, target in tqdm(train_loader):
            data, target = (
                data.to(global_torch_device()),
                target.to(global_torch_device()),
            )
            optimizer.zero_grad()
            output, *_ = model(data)
            output = torch.sigmoid(output)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        with TorchEvalSession(model):
            with torch.no_grad():
                for data, target in tqdm(valid_loader):
                    data, target = (
                        data.to(global_torch_device()),
                        target.to(global_torch_device()),
                    )
                    output, *_ = model(
                        data
                    )  # forward pass: compute predicted outputs by passing inputs to the model
                    output = torch.sigmoid(output)
                    loss = criterion(output, target)  # calculate the batch loss
                    valid_loss += loss.item() * data.size(
                        0
                    )  # update average validation loss
                    dice_cof = intersection_over_union(
                        output.cpu().detach().numpy(), target.cpu().detach().numpy()
                    )
                    dice_score += dice_cof * data.size(0)

        # calculate average losses
        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)
        dice_score = dice_score / len(valid_loader.dataset)

        # print training/validation statistics
        E.set_description(
            f"Epoch: {epoch}"
            f" Training Loss: {train_loss:.6f} "
            f"Validation Loss: {valid_loss:.6f} "
            f"Dice Score: {dice_score:.6f}"
        )

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print(
                f"Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}).  Saving model ..."
            )
            torch.save(model.state_dict(), save_model_path)
            valid_loss_min = valid_loss

        scheduler.step()
        model, scheduler = reschedule(model, epoch, scheduler)

    return model


def main():
    pyplot.style.use("bmh")

    base_path = Path.home() / "Data" / "Datasets" / "PennFudanPed"

    save_model_path = PROJECT_APP_PATH.user_data / "penn_fudan_ped_seg.model"
    train_a = False
    eval_a = not train_a
    SEED = 87539842
    batch_size = 8
    num_workers = 1  # os.cpu_count()
    learning_rate = 0.01
    torch_seed(SEED)

    train_set = PennFudanDataset(base_path, Split.Training)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    valid_loader = DataLoader(
        PennFudanDataset(base_path, Split.Validation),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = SkipHourglassFission(
        input_channels=train_set.predictor_shape[-1],
        output_heads=(train_set.response_shape[-1],),
        encoding_depth=1,
    )
    model.to(global_torch_device())

    if train_a:
        if save_model_path.exists():
            model.load_state_dict(torch.load(str(save_model_path)))
            print("loading saved model")

        with TorchTrainSession(model):
            criterion = BCEDiceLoss(eps=1.0)
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 7, eta_min=learning_rate / 100, last_epoch=-1
            )

            model = train_segmenter(
                model,
                train_loader,
                valid_loader,
                criterion,
                optimizer,
                scheduler,
                save_model_path,
            )

    if eval_a:
        if save_model_path.exists():
            model.load_state_dict(torch.load(str(save_model_path)))
            print("loading saved model")

        with torch.no_grad():
            with TorchCacheSession():
                with TorchEvalSession(model):
                    valid_masks = []
                    a = (350, 525)
                    tr = min(len(valid_loader.dataset) * 4, 2000)
                    probabilities = numpy.zeros((tr, *a), dtype=numpy.float32)
                    for sample_i, (data, target) in enumerate(tqdm(valid_loader)):
                        data = data.to(global_torch_device())
                        target = target.cpu().detach().numpy()
                        outpu, *_ = model(data)
                        outpu = torch.sigmoid(outpu).cpu().detach().numpy()
                        for p in range(data.shape[0]):
                            output, mask = outpu[p], target[p]
                            for m in mask:
                                valid_masks.append(cv2_resize(m, a))
                            for probability in output:
                                probabilities[sample_i, :, :] = cv2_resize(
                                    probability, a
                                )
                                sample_i += 1
                            if sample_i >= tr - 1:
                                break
                        if sample_i >= tr - 1:
                            break

                    pyplot.imshow(probabilities[0])
                    pyplot.show()
                    pyplot.imshow(valid_masks[0])
                    pyplot.show()


if __name__ == "__main__":
    main()
