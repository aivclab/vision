# coding: utf-8

__doc__ = r"""
 One Shot Learning with Siamese Networks
"""

import math
from itertools import count
from pathlib import Path

import numpy
import torch
import torch.nn.functional as F
import torchvision.utils
from draugr.numpy_utilities import Split
from draugr.stopping import IgnoreInterruptSignal
from draugr.torch_utilities import (
    TensorBoardPytorchWriter,
    TorchEvalSession,
    TorchTrainSession,
    global_torch_device,
    load_model_parameters,
    save_model_parameters,
    to_tensor,
)
from draugr.writers import MockWriter, Writer
from torch import optim
from torch.nn import TripletMarginLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from neodroidvision import PROJECT_APP_PATH
from neodroidvision.data.classification.nlet import PairDataset, TripletDataset
from neodroidvision.regression import NLetConvNet
from neodroidvision.utilities.visualisation.similarity_utilities import (
    boxed_text_overlay_plot,
)


def accuracy(*, distances, is_diff, threshold: float = 0.5):
    """

    :param distances:
    :type distances:
    :param is_diff:
    :type is_diff:
    :param threshold:
    :type threshold:
    :return:
    :rtype:"""
    return torch.mean(
        (
            is_diff
            == to_tensor(
                distances < threshold, dtype=torch.long, device=global_torch_device()
            )
        ).to(dtype=torch.float)
    )


def vis(model, data_dir, img_size):
    """ """
    # ## Visualising some of the data
    # The top row and the bottom row of any column is one pair. The 0s and 1s correspond to the column of the
    # image.
    # 0 indicates dissimilar, and 1 indicates similar.

    example_batch = next(
        iter(
            DataLoader(
                PairDataset(
                    data_path=data_dir,
                    transform=transforms.Compose(
                        [
                            transforms.Grayscale(),
                            transforms.Resize(img_size),
                            transforms.ToTensor(),
                        ]
                    ),
                    split=Split.Validation,
                ),
                shuffle=True,
                num_workers=0,
                batch_size=8,
            )
        )
    )
    concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
    boxed_text_overlay_plot(
        torchvision.utils.make_grid(concatenated), str(example_batch[2].numpy())
    )


def stest_one_versus_many(model, data_dir, img_size):
    """ """
    data_iterator = iter(
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
                split=Split.Testing,
            ),
            num_workers=0,
            batch_size=1,
            shuffle=True,
        )
    )
    x0, *_ = next(data_iterator)
    for i in range(10):
        _, x1, _ = next(data_iterator)
        dis = (
            torch.pairwise_distance(
                *model(
                    to_tensor(x0, device=global_torch_device()),
                    to_tensor(x1, device=global_torch_device()),
                )
            )
            .cpu()
            .item()
        )
        boxed_text_overlay_plot(
            torchvision.utils.make_grid(torch.cat((x0, x1), 0)),
            f"Dissimilarity: {dis:.2f}",
        )


def stest_many_versus_many(model, data_dir, img_size, threshold=0.5):
    """ """
    data_iterator = iter(
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
            ),
            num_workers=0,
            batch_size=1,
            shuffle=True,
        )
    )
    for i in range(10):
        x0, x1, is_diff = next(data_iterator)
        distance = (
            torch.pairwise_distance(
                *model(
                    to_tensor(x0, device=global_torch_device()),
                    to_tensor(x1, device=global_torch_device()),
                )
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


def train_siamese(
    model,
    optimiser,
    criterion,
    *,
    writer: Writer = MockWriter(),
    train_number_epochs,
    data_dir,
    train_batch_size,
    model_name,
    save_path,
    save_best=False,
    img_size,
    validation_interval: int = 1,
):
    """
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
    :rtype:

      Parameters
      ----------
      img_size
      validation_interval"""

    train_dataloader = DataLoader(
        TripletDataset(
            data_path=data_dir,
            transform=transforms.Compose(
                [
                    transforms.Grayscale(),
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                ]
            ),
            split=Split.Training,
        ),
        shuffle=True,
        num_workers=0,
        batch_size=train_batch_size,
    )

    valid_dataloader = DataLoader(
        TripletDataset(
            data_path=data_dir,
            transform=transforms.Compose(
                [
                    transforms.Grayscale(),
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                ]
            ),
            split=Split.Validation,
        ),
        shuffle=True,
        num_workers=0,
        batch_size=train_batch_size,
    )

    best = math.inf

    E = tqdm(range(0, train_number_epochs))
    batch_counter = count()

    for epoch in E:
        for tss in train_dataloader:
            batch_i = next(batch_counter)
            with TorchTrainSession(model):
                optimiser.zero_grad()
                loss_contrastive = criterion(
                    *model(*[t.to(global_torch_device()) for t in tss])
                )
                loss_contrastive.backward()
                optimiser.step()
                a = loss_contrastive.cpu().item()
                writer.scalar("train_loss", a, batch_i)
            if batch_counter.__next__() % validation_interval == 0:
                with TorchEvalSession(model):
                    for tsv in valid_dataloader:
                        o = model(*[t.to(global_torch_device()) for t in tsv])
                        a_v = criterion(*o).cpu().item()
                        valid_positive_acc = (
                            accuracy(
                                distances=F.pairwise_distance(o[0], o[1]), is_diff=0
                            )
                            .cpu()
                            .item()
                        )
                        valid_negative_acc = (
                            accuracy(
                                distances=F.pairwise_distance(o[0], o[2]), is_diff=1
                            )
                            .cpu()
                            .item()
                        )
                        valid_acc = numpy.mean((valid_negative_acc, valid_positive_acc))
                        writer.scalar("valid_loss", a_v, batch_i)
                        writer.scalar("valid_positive_acc", valid_positive_acc, batch_i)
                        writer.scalar("valid_negative_acc", valid_negative_acc, batch_i)
                        writer.scalar("valid_acc", valid_acc, batch_i)
                        if a_v < best:
                            best = a_v
                            print(f"new best {best}")
                            if save_best:
                                save_model_parameters(
                                    model,
                                    optimiser=optimiser,
                                    model_name=model_name,
                                    save_directory=save_path,
                                )
            E.set_description(
                f"Epoch number {epoch}, Current train loss {a}, valid loss {a_v}, valid acc {valid_acc}"
            )

    return model


if __name__ == "__main__":

    def main():
        """ """
        data_dir = Path.home() / "Data" / "mnist_png"
        train_batch_size = 64
        train_number_epochs = 100
        save_path = PROJECT_APP_PATH.user_data / "models"
        model_name = "triplet_siamese_mnist"
        load_prev = True
        train = False

        img_size = (28, 28)
        model = NLetConvNet(img_size).to(global_torch_device())
        optimiser = optim.Adam(model.parameters(), lr=3e-4)

        if train:
            if load_prev:
                model, optimiser = load_model_parameters(
                    model,
                    optimiser=optimiser,
                    model_name=model_name,
                    model_directory=save_path,
                )

            with TensorBoardPytorchWriter():
                # from draugr.stopping import CaptureEarlyStop

                # with CaptureEarlyStop() as _:
                with IgnoreInterruptSignal():
                    model = train_siamese(
                        model,
                        optimiser,
                        TripletMarginLoss().to(global_torch_device()),
                        train_number_epochs=train_number_epochs,
                        data_dir=data_dir,
                        train_batch_size=train_batch_size,
                        model_name=model_name,
                        save_path=save_path,
                        img_size=img_size,
                    )
            save_model_parameters(
                model,
                optimiser=optimiser,
                model_name=f"{model_name}",
                save_directory=save_path,
            )
        else:
            model = load_model_parameters(
                model, model_name=model_name, model_directory=save_path
            )
            print("loaded best val")
            stest_many_versus_many(model, data_dir, img_size)

    main()
