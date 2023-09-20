#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Christian Heider Nielsen"
__doc__ = r"""
  Training for BetaVae's

  Distangled Representations
"""

import os
import time
import torch
import torch.utils.data
from draugr import max_frequency
from draugr.torch_utilities import (
    TensorBoardPytorchWriter,
    TorchEvalSession,
    TorchTrainSession,
    global_torch_device,
)
from draugr.visualisation import progress_bar
from draugr.writers import Writer
from math import inf
from neodroidvision import PROJECT_APP_PATH
from neodroidvision.regression.vae.architectures.disentangled.beta_vae import (
    HigginsBetaVae,
)
from neodroidvision.regression.vae.architectures.vae import VAE
from neodroidvision.utilities import scatter_plot_encoding_space
from objectives import loss_function
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import Grayscale
from torchvision.utils import save_image

torch.manual_seed(82375329)
LOWEST_L = inf

core_count = 0  # min(8, multiprocessing.cpu_count() - 1)
INPUT_SIZE = 64
CHANNELS = 3

BATCH_SIZE = 64
MODEL_CHECKPOINT_SUFFIX = ".pth"
LR = 3e-4
BETA = 4
ENCODING_SIZE = 2
name = "mnist"
# ORI_INPUT_SIZE = 28
transform = transforms.Compose(
    [
        Grayscale(3),
        transforms.Resize(INPUT_SIZE),
        # transforms.Normalize(),
        transforms.ToTensor(),
    ]
)
inverse_transform = transforms.Compose(
    [
        Grayscale(1),
        # transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist()),
        transforms.ToPILImage(),
    ]
)

DATASET = MNIST(
    str(Path.home() / "Data" / "Datasets" / name),
    train=True,
    download=True,
    transform=transform,
)
GLOBAL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DL_KWARGS = (
    {"num_workers": core_count, "pin_memory": True} if torch.cuda.is_available() else {}
)
BASE_PATH = PROJECT_APP_PATH.user_data / name / "bvae"
if not BASE_PATH.exists():
    BASE_PATH.mkdir(parents=True)

MODEL: VAE = HigginsBetaVae(CHANNELS, latent_size=ENCODING_SIZE).to(
    global_torch_device()
)


def train_model(
    model,
    optimiser,
    epoch_i: int,
    metric_writer: Writer,
    loader: DataLoader,
    log_interval=10,
):
    """

    Args:
      model:
      optimiser:
      epoch_i:
      metric_writer:
      loader:
      log_interval:
    """
    with TorchTrainSession(model):
        train_accum_loss = 0
        generator = progress_bar(enumerate(loader))
        for batch_idx, (original, *_) in generator:
            original = original.to(global_torch_device())

            reconstruction, mean, log_var = model(original)
            loss = loss_function(reconstruction, original, mean, log_var)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            train_accum_loss += loss.item()
            metric_writer.scalar("train_loss", loss.item())

            if batch_idx % log_interval == 0:
                generator.set_description(
                    f"Train Epoch: {epoch_i}"
                    f" [{batch_idx * len(original)}/"
                    f"{len(loader.dataset)}"
                    f" ({100. * batch_idx / len(loader):.0f}%)]\t"
                    f"Loss: {loss.item() / len(original):.6f}"
                )
            break
        print(
            f"====> Epoch: {epoch_i}"
            f" Average loss: {train_accum_loss / len(loader.dataset):.4f}"
        )


def stest_model(
    model: VAE,
    epoch_i: int,
    metric_writer: Writer,
    loader: DataLoader,
    save_images: bool = True,
):
    """

    Args:
      model:
      epoch_i:
      metric_writer:
      loader:
      save_images:
    """
    global LOWEST_L
    with TorchEvalSession(model):
        test_accum_loss = 0

        with torch.no_grad():
            for i, (original, labels, *_) in enumerate(loader):
                original = original.to(global_torch_device())

                reconstruction, mean, log_var = model(original)
                loss = loss_function(reconstruction, original, mean, log_var).item()

                test_accum_loss += loss
                metric_writer.scalar("test_loss", test_accum_loss)

                if save_images and max_frequency("save_i", 100):
                    if i == 0:
                        n = min(original.size(0), 8)
                        comparison = torch.cat([original[:n], reconstruction[:n]])
                        save_image(
                            comparison.cpu(),  # Torch save images
                            str(BASE_PATH / f"reconstruction_{str(epoch_i)}.png"),
                            nrow=n,
                        )

                    scatter_plot_encoding_space(
                        str(BASE_PATH / f"encoding_space_{str(epoch_i)}.png"),
                        mean.to("cpu").numpy(),
                        log_var.to("cpu").numpy(),
                        labels,
                    )

                break

        # test_loss /= len(loader.dataset)
        test_accum_loss /= loader.batch_size
        print(f"====> Test set loss: {test_accum_loss:.4f}")
        torch.save(
            model.state_dict(),
            BASE_PATH / f"model_state_dict{str(epoch_i)}{MODEL_CHECKPOINT_SUFFIX}",
        )

        if LOWEST_L > test_accum_loss:
            if max_frequency("model_save", 5):
                LOWEST_L = test_accum_loss
                torch.save(
                    model.state_dict(),
                    BASE_PATH / f"best_state_dict{MODEL_CHECKPOINT_SUFFIX}",
                )


if __name__ == "__main__":

    def main(*, EPOCHS=1000, training=False, continue_t=True):
        """

        :param EPOCHS:
        :type EPOCHS:
        :param training:
        :type training:
        :param continue_t:
        :type continue_t:
        """
        if continue_t:
            try:
                _list_of_files = list(BASE_PATH.rglob(f"*{MODEL_CHECKPOINT_SUFFIX}"))
                latest_model_path = str(max(_list_of_files, key=os.path.getctime))
                print(f"loading previous model: {latest_model_path}")
                if latest_model_path is not None:
                    MODEL.load_state_dict(torch.load(latest_model_path))
            except:
                pass

        dataset_loader = DataLoader(
            DATASET, batch_size=BATCH_SIZE, shuffle=True, **DL_KWARGS
        )

        optimiser = optim.Adam(MODEL.parameters(), lr=LR, betas=(0.9, 0.999))

        with TensorBoardPytorchWriter(
            PROJECT_APP_PATH.user_log / name / f"{time.time()}"
        ) as metric_writer:
            for ith_epoch in range(1, EPOCHS + 1):
                if training:
                    train_model(
                        MODEL, optimiser, ith_epoch, metric_writer, dataset_loader
                    )

                if max_frequency("test_model", 100):
                    stest_model(MODEL, ith_epoch, metric_writer, dataset_loader)

                if max_frequency("sample_i", 100):
                    with torch.no_grad():
                        inv_sample = inverse_transform(
                            MODEL.sample().view(CHANNELS, INPUT_SIZE, INPUT_SIZE)
                        )
                        inv_sample.save(str(BASE_PATH / f"sample_{str(ith_epoch)}.png"))
                        if ENCODING_SIZE == 2:
                            from neodroidvision.utilities import plot_manifold

                            plot_manifold(
                                MODEL.decoder,
                                out_path=BASE_PATH / f"manifold_{str(ith_epoch)}.png",
                                img_w=INPUT_SIZE,
                                img_h=INPUT_SIZE,
                            )

    main(training=True, continue_t=True)
