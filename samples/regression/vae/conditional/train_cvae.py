#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/03/2020
           """

import os
import time
from collections import defaultdict
from math import inf

import pandas
import seaborn
import torch
from draugr.torch_utilities import ImprovementDetector, global_torch_device
from draugr.visualisation import progress_bar
from matplotlib import pyplot
from neodroidvision import PROJECT_APP_PATH
from neodroidvision.regression.vae.architectures.disentangled.conditional_vae import (
    ConditionalVAE,
)
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from warg import NOD, ensure_existence

from objectives import loss_fn


def main(config, model, tmsp_path, patience=100):
    """description"""
    data_loader = DataLoader(
        dataset=DATASET, batch_size=config.batch_size, shuffle=True
    )

    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    logs = defaultdict(list)

    with ImprovementDetector(patience) as is_improving:
        for epoch_i in progress_bar(range(config.epochs)):
            tracker_epoch = defaultdict(lambda: defaultdict(dict))

            for iteration, (original, label) in progress_bar(enumerate(data_loader)):
                original, label = (
                    original.to(global_torch_device()),
                    label.to(global_torch_device()),
                )
                reconstruction, mean, log_var, z = model(
                    original, one_hot(label, 10).to(GLOBAL_DEVICE)
                )

                for i, yi in enumerate(label):
                    id = len(tracker_epoch)
                    tracker_epoch[id]["x"] = z[i, 0].item()
                    tracker_epoch[id]["y"] = z[i, 1].item()
                    tracker_epoch[id]["label"] = yi.item()

                optimiser.zero_grad()
                loss = loss_fn(reconstruction, original, mean, log_var)
                loss.backward()
                optimiser.step()

                logs["loss"].append(loss.item())

                if not is_improving(loss):
                    break

                if (
                    iteration % config.print_every == 0
                    or iteration == len(data_loader) - 1
                ):
                    print(
                        f"Epoch {epoch_i:02d}/{config.epochs:02d}"
                        f" Batch {iteration:04d}/{len(data_loader) - 1:d},"
                        f" Loss {loss.item():9.4f}"
                    )

                    condition_vector = torch.arange(0, 10, device=GLOBAL_DEVICE).long()
                    sample = model.sample(
                        one_hot(condition_vector, 10).to(GLOBAL_DEVICE),
                        num=condition_vector.size(0),
                    )

                    pyplot.figure()
                    pyplot.figure(figsize=(5, 10))
                    for p in range(10):
                        pyplot.subplot(5, 2, p + 1)

                        pyplot.text(
                            0,
                            0,
                            f"c={condition_vector[p].item():d}",
                            color="black",
                            backgroundcolor="white",
                            fontsize=8,
                        )
                        pyplot.imshow(sample[p].cpu().data.numpy())
                        pyplot.axis("off")

                    pyplot.savefig(
                        str(tmsp_path / f"Epoch{epoch_i:d}_Iter{iteration:d}.png"),
                        dpi=300,
                    )
                    pyplot.clf()
                    pyplot.close("all")

            df = pandas.DataFrame.from_dict(tracker_epoch, orient="index")
            g = seaborn.lmplot(
                x="x",
                y="y",
                hue="label",
                data=df.groupby("label").head(100),
                fit_reg=False,
                legend=True,
            )
            g.savefig(
                str(tmsp_path / f"Epoch{epoch_i:d}_latent_space.png"),
                dpi=300,
            )
            if True:
                torch.save(
                    model.state_dict(),
                    BASE_PATH / f"model_state_dict{str(epoch_i)}.pth",
                )

            # if False and LOWEST_L > test_accum_loss:
            #    LOWEST_L = test_accum_loss
            #    torch.save(model.state_dict(), BASE_PATH / f"best_state_dict.pth")


if __name__ == "__main__":
    CONFIG = NOD()
    CONFIG.seed = 58329583
    CONFIG.epochs = 1000
    CONFIG.batch_size = 256
    CONFIG.learning_rate = 0.001
    CONFIG.encoder_layer_sizes = [784, 256]
    CONFIG.decoder_layer_sizes = [256, 784]
    CONFIG.latent_size = 10
    CONFIG.print_every = 100
    GLOBAL_DEVICE = global_torch_device()
    TIMESTAMP = time.time()

    LOWEST_L = inf

    CORE_COUNT = 0  # min(8, multiprocessing.cpu_count() - 1)

    GLOBAL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DL_KWARGS = (
        {"num_workers": CORE_COUNT, "pin_memory": True}
        if torch.cuda.is_available()
        else {}
    )
    BASE_PATH = ensure_existence(PROJECT_APP_PATH.user_data / "cvae")

    torch.manual_seed(CONFIG.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CONFIG.seed)

    NAME = "MNIST"
    MODEL = ConditionalVAE(
        encoder_layer_sizes=CONFIG.encoder_layer_sizes,
        latent_size=CONFIG.latent_size,
        decoder_layer_sizes=CONFIG.decoder_layer_sizes,
        num_conditions=10,
    ).to(global_torch_device())
    DATASET = MNIST(
        root=str(PROJECT_APP_PATH.user_data / NAME),
        train=True,
        transform=transforms.ToTensor(),
        download=True,
    )

    USER_DATA_PATH = PROJECT_APP_PATH.user_data / NAME / "cvae"
    TMSP_PATH = USER_DATA_PATH / str(TIMESTAMP)
    if not TMSP_PATH.exists():
        TMSP_PATH.mkdir(parents=True)

    if True:
        _list_of_files = list(USER_DATA_PATH.rglob("*.pth"))
        _latest_model_path = str(max(_list_of_files, key=os.path.getctime))
        print(f"loading previous model: {_latest_model_path}")
        if _latest_model_path is not None:
            MODEL.load_state_dict(torch.load(_latest_model_path))

    main(CONFIG, MODEL, TMSP_PATH)
