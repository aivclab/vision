#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/03/2020
           """

import os
import time
from collections import defaultdict

import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot
from neodroidvision import PROJECT_APP_PATH
from neodroidvision.regression.vae.architectures.conditional_vae import ConditionalVAE
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from warg import NOD

from draugr.torch_utilities import global_torch_device
from .objectives import loss_fn

fig_root = PROJECT_APP_PATH.user_data / "cvae"

config = NOD()
config.seed = 58329583
config.epochs = 1000
config.batch_size = 256
config.learning_rate = 0.001
config.encoder_layer_sizes = [784, 256]
config.decoder_layer_sizes = [256, 784]
config.latent_size = 10
config.print_every = 100
GLOBAL_DEVICE = global_torch_device()
timstamp = time.time()
torch.manual_seed(config.seed)
if torch.cuda.is_available():
  torch.cuda.manual_seed(config.seed)

vae = ConditionalVAE(
    encoder_layer_sizes=config.encoder_layer_sizes,
    latent_size=config.latent_size,
    decoder_layer_sizes=config.decoder_layer_sizes,
    num_conditions=10,
    ).to(global_torch_device())
dataset = MNIST(
    root=str(PROJECT_APP_PATH.user_data / "MNIST"),
    train=True,
    transform=transforms.ToTensor(),
    download=True,
    )


def one_hot(labels, num_labels, device="cpu"):
  targets = torch.zeros(labels.size(0), num_labels)
  for i, label in enumerate(labels):
    targets[i, label] = 1
  return targets.to(device=device)


def main():
  data_loader = DataLoader(
      dataset=dataset, batch_size=config.batch_size, shuffle=True
      )

  optimizer = torch.optim.Adam(vae.parameters(), lr=config.learning_rate)

  logs = defaultdict(list)

  for epoch in range(config.epochs):
    tracker_epoch = defaultdict(lambda:defaultdict(dict))

    for iteration, (original, label) in enumerate(data_loader):

      original, label = (
          original.to(global_torch_device()),
          label.to(global_torch_device()),
          )
      reconstruction, mean, log_var, z = vae(
          original, one_hot(label, 10, device=GLOBAL_DEVICE)
          )

      for i, yi in enumerate(label):
        id = len(tracker_epoch)
        tracker_epoch[id]["x"] = z[i, 0].item()
        tracker_epoch[id]["y"] = z[i, 1].item()
        tracker_epoch[id]["label"] = yi.item()

      optimizer.zero_grad()
      loss = loss_fn(reconstruction, original, mean, log_var)
      loss.backward()
      optimizer.step()

      logs["loss"].append(loss.item())

      if iteration % config.print_every == 0 or iteration == len(data_loader) - 1:
        print(
            f"Epoch {epoch:02d}/{config.epochs:02d}"
            f" Batch {iteration:04d}/{len(data_loader) - 1:d},"
            f" Loss {loss.item():9.4f}"
            )

        condition_vector = (
            torch.arange(0, 10, device=GLOBAL_DEVICE).long().unsqueeze(1)
        )
        sample = vae.sample(
            one_hot(condition_vector, 10, device=GLOBAL_DEVICE),
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

        if not os.path.exists(os.path.join(fig_root, str(timstamp))):
          if not (os.path.exists(os.path.join(fig_root))):
            os.mkdir(os.path.join(fig_root))
          os.mkdir(os.path.join(fig_root, str(timstamp)))

        pyplot.savefig(
            os.path.join(
                fig_root, str(timstamp), f"Epoch{epoch:d}_Iter{iteration:d}.png"
                ),
            dpi=300,
            )
        pyplot.clf()
        pyplot.close("all")

    df = pd.DataFrame.from_dict(tracker_epoch, orient="index")
    g = sns.lmplot(
        x="x",
        y="y",
        hue="label",
        data=df.groupby("label").head(100),
        fit_reg=False,
        legend=True,
        )
    g.savefig(
        os.path.join(fig_root, str(timstamp), f"Epoch{epoch:d}_latent_space.png"),
        dpi=300,
        )


if __name__ == "__main__":

  main()
