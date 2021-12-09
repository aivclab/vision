#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import os
import time

from apppath import ensure_existence
from draugr.numpy_utilities import Split
from draugr.torch_utilities import (
    TensorBoardPytorchWriter,
    TorchEvalSession, global_torch_device,
    to_device_iterator,
)
from draugr.visualisation import plot_img_array, plot_side_by_side
from draugr.writers import Writer
from itertools import cycle
from matplotlib import pyplot
from pathlib import Path
from torch.nn.modules.module import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Iterator

from neodroidvision import PROJECT_APP_PATH
from neodroidvision.multitask import SkipHourglassFission
from neodroidvision.utilities.torch_utilities.layers.torch_layers import MinMaxNorm

__author__ = "Christian Heider Nielsen"

__doc__ = r"""Denoising Example
"""

import torch
from torch import optim
from tqdm import tqdm
from torchvision.datasets import MNIST

criterion = torch.nn.MSELoss()


def training(
        model: Module,
        data_iterator: Iterator,
        optimizer: Optimizer,
        scheduler,
        writer: Writer,
        interrupted_path: Path,
        *,
        num_updates=2500000,
        early_stop_threshold=1e-9,
        denoise: bool = True,
) -> Module:
    """

    :param model:
    :type model:
    :param data_iterator:
    :type data_iterator:
    :param optimizer:
    :type optimizer:
    :param scheduler:
    :type scheduler:
    :param writer:
    :type writer:
    :param interrupted_path:
    :type interrupted_path:
    :param num_updates:
    :type num_updates:
    :param early_stop_threshold:
    :type early_stop_threshold:
    :return:
    :rtype:"""
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    since = time.time()
    # reraser =RandomErasing()

    try:
        sess = tqdm(range(num_updates), leave=False, disable=False)
        for update_i in sess:
            for phase in [Split.Training, Split.Validation]:
                if phase == Split.Training:

                    for param_group in optimizer.param_groups:
                        writer.scalar("lr", param_group["lr"], update_i)

                    model.train()
                else:
                    model.eval()

                rgb_imgs, *_ = next(data_iterator)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == Split.Training):
                    if denoise:  # =='denoise':
                        model_input = rgb_imgs + torch.normal(
                            mean=0.0,
                            std=0.1,
                            size=rgb_imgs.shape,
                            device=global_torch_device(),
                        )
                    # elif recover_type=='missing':
                    #  model_input = rgb_imgs
                    else:
                        model_input = rgb_imgs
                    recon_pred, *_ = model(torch.clamp(model_input, 0.0, 1.0))
                    ret = criterion(recon_pred, rgb_imgs)

                    if phase == Split.Training:
                        ret.backward()
                        optimizer.step()
                        scheduler.step()

                update_loss = ret.data.cpu().numpy()
                writer.scalar(f"loss/accum", update_loss, update_i)

                if phase == Split.Validation and update_loss < best_loss:
                    best_loss = update_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    _format = "NCHW"
                    writer.image(f"rgb_imgs", rgb_imgs, update_i, data_formats=_format)
                    writer.image(
                        f"recon_pred", recon_pred, update_i, data_formats=_format
                    )
                    sess.write(f"New best model at update {update_i}")

            sess.set_description_str(
                f"Update {update_i} - {phase} accum_loss:{update_loss:2f}"
            )

            if update_loss < early_stop_threshold:
                break
    except KeyboardInterrupt:
        print("Interrupt")
    finally:
        model.load_state_dict(best_model_wts)  # load best model weights
        torch.save(model.state_dict(), interrupted_path)

    time_elapsed = time.time() - since
    print(f"{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val loss: {best_loss}")

    return model


def inference(model: Module, data_iterator: Iterator, denoise: bool = True, num: int = 3) -> None:
    """

    :param model:
    :type model:
    :param data_iterator:
    :type data_iterator:"""
    with torch.no_grad():
        with TorchEvalSession(model):
            img, target = next(data_iterator)
            if denoise:
                model_input = img + torch.normal(
                    mean=0.0, std=0.1, size=img.shape, device=global_torch_device()
                )
            else:
                model_input = img
            model_input = torch.clamp(model_input, 0.0, 1.0)
            pred, *_ = model(model_input)
            plot_side_by_side([pred.squeeze(1)[:num].cpu().numpy(),
                               model_input.squeeze(1)[:num].cpu().numpy()])
            pyplot.show()
            return
            for i, (s, j, label) in enumerate(
                    zip(pred.cpu().numpy(), model_input.cpu().numpy(), target)
            ):
                pyplot.imshow(j[0])
                pyplot.title(f'sample_{i}, category: {label}')
                pyplot.show()
                # plot_side_by_side(s)
                pyplot.imshow(s[0])
                pyplot.title(f'sample_{i}, category: {label}')
                pyplot.show()
                break


def train_mnist(load_earlier=False, train=True, denoise: bool = True):
    """

    :param load_earlier:
    :type load_earlier:
    :param train:
    :type train:"""
    seed = 251645
    batch_size = 32

    tqdm.monitor_interval = 0
    learning_rate = 3e-3
    lr_sch_step_size = int(10e4 // batch_size)
    lr_sch_gamma = 0.1
    unet_depth = 3
    unet_start_channels = 16
    input_channels = 1
    output_channels = (input_channels,)

    home_path = PROJECT_APP_PATH
    model_file_ending = ".model"
    model_base_path = ensure_existence(PROJECT_APP_PATH.user_data / "unet_mnist")
    interrupted_name = 'INTERRUPTED_BEST'
    interrupted_path = (
            model_base_path / f"{interrupted_name}{model_file_ending}"
    )

    torch.manual_seed(seed)

    device = global_torch_device()

    img_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            MinMaxNorm(),
            transforms.Lambda(lambda tensor: torch.round(tensor)),
            # transforms.RandomErasing()
        ]
    )
    dataset = MNIST(
        PROJECT_APP_PATH.user_data / "mnist", transform=img_transform, download=True
    )
    data_iter = iter(
        cycle(DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True))
    )
    data_iter = to_device_iterator(data_iter, device)

    model = SkipHourglassFission(
        input_channels=input_channels,
        output_heads=output_channels,
        encoding_depth=unet_depth,
        start_channels=unet_start_channels,
    ).to(global_torch_device())

    optimizer_ft = optim.Adam(model.parameters(), lr=learning_rate)

    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer_ft, step_size=lr_sch_step_size, gamma=lr_sch_gamma
    )

    if load_earlier:
        _list_of_files = list(model_base_path.rglob(f"{interrupted_name}{model_file_ending}"))
        if not len(_list_of_files):
            print(f'found no trained models under {model_base_path}/**/{interrupted_name}{model_file_ending}')
            exit(1)
        latest_model_path = str(max(_list_of_files, key=os.path.getctime))
        print(f"loading previous model: {latest_model_path}")
        if latest_model_path is not None:
            model.load_state_dict(torch.load(latest_model_path))

    if train:
        with TensorBoardPytorchWriter(home_path.user_log / str(time.time())) as writer:
            model = training(
                model,
                data_iter,
                optimizer_ft,
                exp_lr_scheduler,
                writer,
                interrupted_path,
                denoise=denoise,
            )
            torch.save(
                model.state_dict(), model_base_path / f"unet_mnist_final{model_file_ending}"
            )
    else:
        inference(model, data_iter, denoise=denoise)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    # train_mnist(load_earlier=False, train=True)
    train_mnist(load_earlier=True, train=False)
