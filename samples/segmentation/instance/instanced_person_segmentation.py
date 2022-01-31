#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from tokenize import Number

import numpy
import torch
from apppath import ensure_existence
from draugr.numpy_utilities import chw_to_hwc, SplitEnum
from draugr.opencv_utilities import cv2_resize
from draugr.random_utilities import seed_stack


from draugr.torch_utilities import (
    TorchCacheSession,
    TorchDeviceSession,
    TorchEvalSession,
    TorchTrainSession,
    global_torch_device,
)

from matplotlib import pyplot
from torch.utils.data import DataLoader
from tqdm import tqdm
from draugr.writers import Writer, MockWriter, ImageWriterMixin
from draugr.torch_utilities import TensorBoardPytorchWriter
from neodroidvision import PROJECT_APP_PATH
from neodroidvision.data.mixed import PennFudanDataset
from neodroidvision.multitask import SkipHourglassFission
from neodroidvision.segmentation import BCEDiceLoss, intersection_over_union
import time

from neodroidvision.segmentation.evaluation.dice_loss import dice_loss

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

            #TODO: HANDLE multi channel output with interchannel communication

           Created on 09/10/2019
           """


def reschedule_learning_rate(
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    epoch: int,
    scheduler: torch.optim.lr_scheduler,
    starting_learning_rate: Number,
    total_epochs=100,
):
    r"""This may be improved its just a hacky way to write SGDWR"""
    #   #TODO: base on total num epochs instead

    if epoch == 7:
        optimiser = torch.optim.SGD(model.parameters(), lr=starting_learning_rate / 2)
        current_lr = next(iter(optimiser.param_groups))["lr"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, 6, eta_min=current_lr / 100, last_epoch=-1
        )
        print(f"Current LR: {current_lr}")
    if epoch == 13:
        optimiser = torch.optim.SGD(
            model.parameters(), lr=starting_learning_rate / 2 ** 2
        )
        current_lr = next(iter(optimiser.param_groups))["lr"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, 6, eta_min=current_lr / 100, last_epoch=-1
        )
        print(f"Current LR: {current_lr}")
    if epoch == 19:
        optimiser = torch.optim.SGD(
            model.parameters(), lr=starting_learning_rate / 2 ** 3
        )
        current_lr = next(iter(optimiser.param_groups))["lr"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, 6, eta_min=current_lr / 100, last_epoch=-1
        )
        print(f"Current LR: {current_lr}")
    if epoch == 25:
        optimiser = torch.optim.SGD(
            model.parameters(), lr=starting_learning_rate / 2 ** 4
        )
        current_lr = next(iter(optimiser.param_groups))["lr"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, 6, eta_min=current_lr / 100, last_epoch=-1
        )
        print(f"Current LR: {current_lr}")
    if epoch == 50:
        optimiser = torch.optim.SGD(
            model.parameters(), lr=starting_learning_rate / 2 ** 5
        )
        current_lr = next(iter(optimiser.param_groups))["lr"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, 6, eta_min=current_lr / 100, last_epoch=-1
        )
        print(f"Current LR: {current_lr}")
    if epoch == 75:
        optimiser = torch.optim.SGD(
            model.parameters(), lr=starting_learning_rate / 2 ** 6
        )
        current_lr = next(iter(optimiser.param_groups))["lr"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, 6, eta_min=current_lr / 100, last_epoch=-1
        )
        print(f"Current LR: {current_lr}")

    return optimiser, scheduler


def train_person_segmentor(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    criterion: callable,
    optimiser: torch.optim.Optimizer,
    *,
    save_model_path: Path,
    learning_rate: Number = 6e-2,
    scheduler: torch.optim.lr_scheduler = None,
    n_epochs: int = 100,
    writer: ImageWriterMixin = MockWriter(),
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
    :param optimiser:
    :type optimiser:
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
    for epoch_i in E:
        train_loss = 0.0
        valid_loss = 0.0

        with TorchTrainSession(model):
            for data, target in tqdm(train_loader):
                output, *_ = model(data.to(global_torch_device()))
                loss = criterion(output, target.to(global_torch_device()).float())

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                train_loss += loss.cpu().item() * data.size(0)

        with TorchEvalSession(model):
            with torch.no_grad():
                for data, target in tqdm(valid_loader):
                    target = target.float()
                    (
                        output,
                        *_,
                    ) = model(  # forward pass: compute predicted outputs by passing inputs to the model
                        data.to(global_torch_device())
                    )
                    validation_loss = criterion(  # calculate the batch loss
                        output, target.to(global_torch_device())
                    )
                    writer.scalar(
                        "dice_validation",
                        dice_loss(output, target.to(global_torch_device())),
                    )

                    valid_loss += validation_loss.detach().cpu().item() * data.size(
                        0
                    )  # update average validation loss
                writer.image("input", data, epoch_i)  # write the last batch
                writer.image("truth", target, epoch_i)  # write the last batch
                writer.image(
                    "prediction", torch.sigmoid(output), epoch_i
                )  # write the last batch

        # calculate average losses
        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print(
                f"Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}).  Saving model ..."
            )
            torch.save(model.state_dict(), save_model_path)
            valid_loss_min = valid_loss

        if scheduler:
            scheduler.step()
            optimiser, scheduler = reschedule_learning_rate(
                model,
                optimiser,
                epoch_i,
                scheduler,
                starting_learning_rate=learning_rate,
            )

        # print training/validation statistics
        current_lr = next(iter(optimiser.param_groups))["lr"]
        E.set_description(
            f"Epoch: {epoch_i} "
            f"Training Loss: {train_loss:.6f} "
            f"Validation Loss: {valid_loss:.6f} "
            f"Learning rate: {current_lr:.6f}"
        )
        writer.scalar("training_loss", train_loss)
        writer.scalar("validation_loss", valid_loss)
        writer.scalar("learning_rate", current_lr)

    return model


def validate_model(model, valid_loader):
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


def main(
    base_path: Path = Path.home() / "Data" / "Datasets" / "PennFudanPed",
    train_model: bool = True,
    load_prev_model: bool = True,
    writer: Writer = TensorBoardPytorchWriter(
        PROJECT_APP_PATH.user_log / "instanced_person_segmentation" / f"{time.time()}"
    ),
):
    """ """

    # base_path = Path("/") / "encrypted_disk" / "heider" / "Data" / "PennFudanPed"
    base_path: Path = Path.home() / "Data3" / "PennFudanPed"
    # base_path = Path('/media/heider/OS/Users/Christian/Data/Datasets/')  / "PennFudanPed"
    pyplot.style.use("bmh")

    save_model_path = (
        ensure_existence(PROJECT_APP_PATH.user_data / "models")
        / "instanced_penn_fudan_ped_seg.model"
    )

    eval_model = not train_model
    SEED = 9221
    batch_size = 32
    num_workers = 0
    encoding_depth = 2
    learning_rate = 6e-6  # sequence 6e-2 6e-3 6e-4 6e-5

    seed_stack(SEED)

    train_set = PennFudanDataset(
        base_path,
        SplitEnum.training,
        return_variant=PennFudanDataset.PennFudanReturnVariantEnum.instanced,
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    valid_loader = DataLoader(
        PennFudanDataset(
            base_path,
            SplitEnum.validation,
            return_variant=PennFudanDataset.PennFudanReturnVariantEnum.instanced,
        ),
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

    if load_prev_model and save_model_path.exists():
        model.load_state_dict(torch.load(str(save_model_path)))
        print("loading saved model")

    if train_model:
        with TorchTrainSession(model):
            criterion = BCEDiceLoss()
            # optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)
            optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(                optimiser, T_max=7, eta_min=learning_rate / 100, last_epoch=-1            )

            model = train_person_segmentor(
                model,
                train_loader,
                valid_loader,
                criterion,
                optimiser,
                save_model_path=save_model_path,
                learning_rate=learning_rate,
                writer=writer,
            )

    if eval_model:
        validate_model(model, valid_loader)


if __name__ == "__main__":
    main()
    # main(train_model=True, load_prev_model=False)
    # main(train_model=False)
