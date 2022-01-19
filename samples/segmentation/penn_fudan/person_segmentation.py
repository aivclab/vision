#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import numpy
import torch
from apppath import ensure_existence
from draugr.numpy_utilities import chw_to_hwc, SplitEnum
from draugr.opencv_utilities import cv2_resize
from draugr.random_utilities import seed_stack

# from draugr.opencv_utilities import cv2_resize
from draugr.torch_utilities import (
    TorchCacheSession,
    TorchDeviceSession,
    TorchEvalSession,
    TorchTrainSession,
    global_torch_device,
)
from matplotlib import pyplot
from torch.optim import optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from neodroidvision import PROJECT_APP_PATH
from neodroidvision.data.segmentation import PennFudanDataset
from neodroidvision.multitask import SkipHourglassFission
from neodroidvision.segmentation import BCEDiceLoss, intersection_over_union

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """


def reschedule_learning_rate(
    model: torch.nn.Module, epoch: int, scheduler: torch.optim.lr_scheduler
):
    r"""This may be improved its just a hacky way to write SGDWR"""
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


def train_person_segmentor(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    criterion: callable,
    optimizer: optimizer,
    scheduler: torch.optim.lr_scheduler,
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
    :rtype:"""
    valid_loss_min = numpy.Inf  # track change in validation loss
    assert n_epochs > 0, n_epochs
    E = tqdm(range(1, n_epochs + 1))
    for epoch in E:
        train_loss = 0.0
        valid_loss = 0.0
        dice_score = 0.0

        with TorchTrainSession(model):
            for data, target in tqdm(train_loader):
                data, target = (
                    data.to(global_torch_device()),
                    target.to(global_torch_device()),
                )
                optimizer.zero_grad()
                output, *_ = model(data)
                loss = criterion(output, target.float())
                loss.backward()
                optimizer.step()
                train_loss += loss.detach().cpu().item() * data.size(0)

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
                    validation_loss = criterion(
                        output, target.float()
                    )  # calculate the batch loss
                    valid_loss += validation_loss.detach().cpu().item() * data.size(
                        0
                    )  # update average validation loss
                    dice_cof = intersection_over_union(
                        torch.sigmoid(output).cpu().detach().numpy(),
                        target.cpu().detach().numpy(),
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
        model, scheduler = reschedule_learning_rate(model, epoch, scheduler)

    return model


def main(
    base_path: Path = Path.home() / "Data" / "Datasets" / "PennFudanPed",
    train_model: bool = False,
    load_prev_model: bool = True,
):
    """ """

    # base_path = Path("/") / "encrypted_disk" / "heider" / "Data" / "PennFudanPed"
    base_path: Path = Path.home() / "Data3" / "PennFudanPed"
    # base_path = Path('/media/heider/OS/Users/Christian/Data/Datasets/')  / "PennFudanPed"
    pyplot.style.use("bmh")

    save_model_path = (
        ensure_existence(PROJECT_APP_PATH.user_data / "models")
        / "penn_fudan_ped_seg.model"
    )

    eval_model = not train_model
    SEED = 87539842
    batch_size = 4
    num_workers = 0
    encoding_depth = 1
    learning_rate = 0.01
    seed_stack(SEED)

    train_set = PennFudanDataset(base_path, SplitEnum.training)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    valid_loader = DataLoader(
        PennFudanDataset(base_path, SplitEnum.validation),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = SkipHourglassFission(
        input_channels=train_set.predictor_shape[-1],
        output_heads=(train_set.response_shape[-1],),
        encoding_depth=encoding_depth,
    )
    model.to(global_torch_device())

    if train_model:
        if load_prev_model and save_model_path.exists():
            model.load_state_dict(torch.load(str(save_model_path)))
            print("loading saved model")

        with TorchTrainSession(model):
            criterion = BCEDiceLoss(
                # eps=1.0,
                # activation=torch.sigmoid
            )
            optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=7, eta_min=learning_rate / 100, last_epoch=-1
            )

            model = train_person_segmentor(
                model,
                train_loader,
                valid_loader,
                criterion,
                optimiser,
                scheduler,
                save_model_path,
            )

    if eval_model:
        if load_prev_model and save_model_path.exists():
            model.load_state_dict(torch.load(str(save_model_path)))
            print("loading saved model")

        with TorchDeviceSession(global_torch_device("cpu"), model):
            with torch.no_grad():
                with TorchCacheSession():
                    with TorchEvalSession(model):
                        valid_masks = []
                        out_data = []
                        a = (256, 256)
                        tr = min(len(valid_loader.dataset) * 4, 2000)
                        probabilities = []
                        for sample_i, (data, target) in enumerate(tqdm(valid_loader)):
                            data = data.to(global_torch_device())
                            outpu, *_ = model(data)
                            for m, d, p in zip(
                                target.cpu().detach().numpy(),
                                data.cpu().detach().numpy(),
                                torch.sigmoid(outpu).cpu().detach().numpy(),
                            ):
                                out_data.append(cv2_resize(chw_to_hwc(d), a))
                                valid_masks.append(cv2_resize(m[0], a))
                                probabilities.append(cv2_resize(p[0], a))
                                sample_i += 1

                                if sample_i >= tr - 1:
                                    break

                            if sample_i >= tr - 1:
                                break

                        min_a = min(3, len(out_data))
                        f, ax = pyplot.subplots(min_a, 3, figsize=(24, 12))

                        # assert len(valid_masks)>2, f'{len(valid_masks), tr}'
                        for i in range(min_a):
                            ax[0, i].imshow(out_data[i], vmin=0, vmax=1)
                            ax[0, i].set_title("Original", fontsize=14)

                            ax[1, i].imshow(valid_masks[i], vmin=0, vmax=1)
                            ax[1, i].set_title("Target", fontsize=14)

                            ax[2, i].imshow(probabilities[i], vmin=0, vmax=1)
                            ax[2, i].set_title("Prediction", fontsize=14)

                        pyplot.show()


if __name__ == "__main__":
    main(train_model=True, load_prev_model=False)
    # main(train_model=False)
