#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Christian Heider Nielsen"
__doc__ = r"""
 One Shot Learning with Siamese Networks
"""

import math
import time
from itertools import count
from pathlib import Path
from statistics import mean
from typing import Tuple

import torch
import torchvision
from draugr.numpy_utilities import SplitEnum
from draugr.torch_utilities import (
    OverfitDetector,
    TensorBoardPytorchWriter,
    TorchEvalSession,
    TorchTrainSession,
    global_torch_device,
    load_model_parameters,
    save_model_parameters,
    to_tensor,
)
from draugr.visualisation import progress_bar
from draugr.writers import MockWriter, Writer
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from warg import IgnoreInterruptSignal

from neodroidvision import PROJECT_APP_PATH
from neodroidvision.data.classification.nlet import PairDataset
from neodroidvision.data.synthesis.conversion.mnist.convert_mnist_to_png import (
    generate_mnist_png,
)
from neodroidvision.regression.metric.contrastive.pair_ranking import PairRankingSiamese
from neodroidvision.utilities.visualisation.similarity_utilities import (
    boxed_text_overlay_plot,
)


def accuracy(
    *, distances: torch.Tensor, is_different: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    """

    :param distances:
    :type distances:
    :param is_different:
    :type is_different:
    :param threshold:
    :type threshold:
    :return:
    :rtype:"""
    return torch.mean(
        (
            is_different
            == to_tensor(
                distances > threshold, dtype=torch.long, device=global_torch_device()
            )
        ).to(dtype=torch.float)
    )


def train_siamese(
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    criterion: callable,
    *,
    writer: Writer = MockWriter(),
    train_number_epochs: int,
    data_dir: Path,
    train_batch_size: int,
    model_name: str,
    save_path: Path,
    save_best: bool = False,
    img_size: Tuple[int, int],
    validation_interval: int = 1,
    patience: int = 10,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    """
    :param patience:
    :type patience:
    :param img_size:
    :type img_size:
    :param validation_interval:
    :type validation_interval:
    :param data_dir:
    :type data_dir:
    :param optimiser:
    :type optimiser:
    :param criterion:
    :type criterion:
    :param writer:
    :type writer:
    :param model_name:
    :type model_name:
    :param save_path:
    :type save_path:
    :param save_best:
    :type save_best:
    :param model:
    :type model:
    :param train_number_epochs:
    :type train_number_epochs:
    :param train_batch_size:
    :type train_batch_size:
    :return:
    :rtype:"""

    train_dataloader = DataLoader(
        PairDataset(
            data_path=data_dir,
            transform=transforms.Compose(
                [
                    transforms.Grayscale(),
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                ]
            ),
            split=SplitEnum.training,
        ),
        shuffle=True,
        num_workers=0,
        batch_size=train_batch_size,
    )

    valid_dataloader = DataLoader(
        PairDataset(
            data_path=data_dir,
            transform=transforms.Compose(
                [
                    transforms.Grayscale(),
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                ]
            ),
            split=SplitEnum.validation,
        ),
        shuffle=True,
        num_workers=0,
        batch_size=train_batch_size,
    )

    best = math.inf

    E = progress_bar(range(0, train_number_epochs))
    batch_counter = count()

    with OverfitDetector(patience) as is_overfitting:
        for epoch in E:
            for tss in train_dataloader:
                batch_i = next(batch_counter)
                with TorchTrainSession(model):
                    o = [t.to(global_torch_device()) for t in tss]
                    optimiser.zero_grad()
                    loss_contrastive = criterion(
                        model(*o[:2]), o[2].to(dtype=torch.float)
                    )
                    loss_contrastive.backward()
                    optimiser.step()
                    train_loss = loss_contrastive.cpu().item()
                    writer.scalar("train_loss", train_loss, batch_i)
                if batch_counter.__next__() % validation_interval == 0:
                    with TorchEvalSession(model):
                        valid_loss = 0
                        valid_accuracy = []
                        for tsv in valid_dataloader:
                            ov = [t.to(global_torch_device()) for t in tsv]
                            v_o, fact = model(*ov[:2]), ov[2].to(dtype=torch.float)
                            valid_loss += criterion(v_o, fact).cpu().item()
                            valid_accuracy += [
                                accuracy(distances=v_o, is_different=fact).cpu().item()
                            ]

                        if is_overfitting(valid_loss):
                            break

                        valid_accuracy = mean(valid_accuracy)
                        writer.scalar("valid_loss", valid_loss, batch_i)
                        if valid_loss < best:
                            best = valid_loss
                            print(f"new best {best}")
                            writer.blip("new_best", batch_i)
                            if save_best:
                                save_model_parameters(
                                    model,
                                    optimiser=optimiser,
                                    model_name=model_name,
                                    save_directory=save_path,
                                )
                E.set_description(
                    f"Epoch number {epoch}, Current train loss {train_loss}, valid loss {valid_loss}, valid_accuracy "
                    f"{valid_accuracy}"
                )

    return model, optimiser


def stest_many_versus_many2(
    model: torch.nn.Module, data_dir: Path, img_size: Tuple[int, int], threshold=0.5
) -> None:
    """

    :param model:
    :type model:
    :param data_dir:
    :type data_dir:
    :param img_size:
    :type img_size:
    :param threshold:
    :type threshold:"""
    dataiter = iter(
        DataLoader(
            PairDataset(
                data_dir,
                transform=transforms.Compose(
                    [
                        transforms.Grayscale(),
                        transforms.Resize(img_size),
                        transforms.ToTensor(),
                    ]
                ),
                return_categories=False,
            ),
            num_workers=0,
            batch_size=1,
            shuffle=True,
        )
    )
    for i in range(10):
        x0, x1, is_diff = next(dataiter)
        distance = (
            model(
                to_tensor(x0, device=global_torch_device()),
                to_tensor(x1, device=global_torch_device()),
            )
            .cpu()
            .item()
        )

        boxed_text_overlay_plot(
            torchvision.utils.make_grid(torch.cat((x0, x1), 0)),
            f"Truth: {'Different' if is_diff.cpu().item() else 'Alike'},"
            f" Dissimilarity: {distance:.2f},"
            f" Verdict: {'Different' if distance > threshold else 'Alike'}",
        )


if __name__ == "__main__":

    def main():
        """description"""
        data_dir = Path.home() / "Data" / "mnist_png"
        train_batch_size = 64
        train_number_epochs = 100
        save_path = PROJECT_APP_PATH.user_data / "models"
        model_name = "pair_siamese_mnist"
        load_previous_model = True
        train = True
        img_size = (28, 28)
        model = PairRankingSiamese(img_size).to(global_torch_device())
        optimiser = optim.Adam(model.parameters(), lr=3e-4)

        if not data_dir.exists():
            # raise FileNotFoundError(f"{data_dir} does not exist")
            generate_mnist_png(data_dir)

        if train:
            if load_previous_model:
                (model, optimiser), _ = load_model_parameters(
                    model,
                    optimiser=optimiser,
                    model_name=model_name,
                    model_directory=save_path,
                )

            with TensorBoardPytorchWriter(
                PROJECT_APP_PATH.user_log / model_name / str(time.time())
            ) as writer:
                # with CaptureEarlyStop() as _:
                with IgnoreInterruptSignal():
                    model, optimiser = train_siamese(
                        model,
                        optimiser,
                        nn.BCELoss().to(global_torch_device()),
                        train_number_epochs=train_number_epochs,
                        data_dir=data_dir,
                        train_batch_size=train_batch_size,
                        model_name=model_name,
                        save_path=save_path,
                        writer=writer,
                        img_size=img_size,
                    )
            save_model_parameters(
                model,
                optimiser=optimiser,
                model_name=f"{model_name}",
                save_directory=save_path,
            )
        else:
            model, _ = load_model_parameters(
                model, model_name=model_name, model_directory=save_path
            )
            print("loaded best val")
            stest_many_versus_many2(model, data_dir, img_size)

    main()
