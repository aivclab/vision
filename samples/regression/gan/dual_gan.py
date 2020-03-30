#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from typing import Iterable

import numpy
import torchvision
from torch import nn

from draugr.torch_utilities import TensorBoardPytorchWriter
from neodroidvision import PROJECT_APP_PATH

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
           """

import torch
import torch.nn

import torch.optim as optim
from torch.autograd import Variable
from itertools import chain

BATCH_SIZE = 32
from sklearn.preprocessing import LabelEncoder

one_hot_encoder = LabelEncoder()
one_hot_encoder.fit(range(10))


def to_one_hot(values):
    value_idxs = one_hot_encoder.transform([values])
    return torch.eye(len(one_hot_encoder.classes_))[value_idxs]


def log(x):
    return torch.log(x + 1e-8)


def reset_grads(modules: Iterable[nn.Module]):
    for m in modules:
        m.zero_grad()


def sample_x(X, size):
    start_idx = numpy.random.randint(0, X.shape[0] - size)
    return X[start_idx : start_idx + size]


def main():
    mnist_l = torchvision.datasets.MNIST(
        PROJECT_APP_PATH.user_cache / "data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
        # target_transform=to_one_hot
    )
    mnist = torch.utils.data.DataLoader(
        mnist_l, batch_size=len(mnist_l.data), shuffle=True
    )

    z_dim = 10
    X_dim = 28 * 28
    h_dim = 128
    cnt = 0
    lr = 1e-4
    n_critics = 3
    lam1, lam2 = 100, 100
    num_samples = 4

    Generator1 = torch.nn.Sequential(
        torch.nn.Linear(X_dim + z_dim, h_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(h_dim, X_dim),
        torch.nn.Sigmoid(),
    )

    Generator2 = torch.nn.Sequential(
        torch.nn.Linear(X_dim + z_dim, h_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(h_dim, X_dim),
        torch.nn.Sigmoid(),
    )

    Discriminator1 = torch.nn.Sequential(
        torch.nn.Linear(X_dim, h_dim), torch.nn.ReLU(), torch.nn.Linear(h_dim, 1)
    )

    Discriminator2 = torch.nn.Sequential(
        torch.nn.Linear(X_dim, h_dim), torch.nn.ReLU(), torch.nn.Linear(h_dim, 1)
    )

    Generators_solver = optim.RMSprop(
        chain(Generator1.parameters(), Generator2.parameters()), lr=lr
    )
    Discriminator1_solver = optim.RMSprop(Discriminator1.parameters(), lr=lr)
    Discriminator2_solver = optim.RMSprop(Discriminator2.parameters(), lr=lr)

    X_train = next(iter(mnist))[0]
    half = int(X_train.shape[0] / 2)

    X_train1 = X_train[:half]

    X_train2: torch.Tensor = X_train[half:]
    X_train2 = X_train2.rot90(dims=(2, 3))

    X_train1 = X_train1.reshape(-1, 28 * 28)
    X_train2 = X_train2.reshape(-1, 28 * 28)

    del X_train  # Cleanup

    with TensorBoardPytorchWriter(
        PROJECT_APP_PATH.user_log / str(time.time())
    ) as writer:

        for it in range(1000000):
            for _ in range(n_critics):
                # Sample data
                z1 = Variable(torch.randn(BATCH_SIZE, z_dim))
                z2 = Variable(torch.randn(BATCH_SIZE, z_dim))
                X1 = sample_x(X_train1, BATCH_SIZE)
                X2 = sample_x(X_train2, BATCH_SIZE)

                # D1
                X2_sample = Generator1(torch.cat([X1, z1], -1))  # G1: X1 -> X2
                D1_real = Discriminator1(X2)
                D1_fake = Discriminator1(X2_sample)

                D1_loss = -(torch.mean(D1_real) - torch.mean(D1_fake))

                D1_loss.backward(retain_graph=True)
                Discriminator1_solver.step()

                # Weight clipping
                for p in Discriminator1.parameters():
                    p.data.clamp_(-0.01, 0.01)

                reset_grads([Generator1, Generator2, Discriminator1, Discriminator2])

                # D2
                X1_sample = Generator2(torch.cat([X2, z2], -1))  # G2: X2 -> X1
                D2_real = Discriminator2(X1)
                D2_fake = Discriminator2(X1_sample)

                D2_loss = -(torch.mean(D2_real) - torch.mean(D2_fake))

                D2_loss.backward()
                Discriminator2_solver.step()

                # Weight clipping
                for p in Discriminator2.parameters():
                    p.data.clamp_(-0.01, 0.01)

                reset_grads([Generator1, Generator2, Discriminator1, Discriminator2])

            # Generator
            z1 = Variable(torch.randn(BATCH_SIZE, z_dim))
            z2 = Variable(torch.randn(BATCH_SIZE, z_dim))
            X1 = sample_x(X_train1, BATCH_SIZE)
            X2 = sample_x(X_train2, BATCH_SIZE)

            X1_sample = Generator2(torch.cat([X2, z2], 1))
            X2_sample = Generator1(torch.cat([X1, z1], 1))

            X1_recon = Generator2(torch.cat([X2_sample, z2], 1))
            X2_recon = Generator1(torch.cat([X1_sample, z1], 1))

            D1_fake = Discriminator1(X1_sample)
            D2_fake = Discriminator2(X2_sample)

            G_loss = -torch.mean(D1_fake) - torch.mean(D2_fake)
            reg1 = lam1 * torch.mean(torch.sum(torch.abs(X1_recon - X1), 1))
            reg2 = lam2 * torch.mean(torch.sum(torch.abs(X2_recon - X2), 1))

            G_loss += reg1 + reg2

            G_loss.backward()
            Generators_solver.step()
            reset_grads([Generator1, Generator2, Discriminator1, Discriminator2])

            if it % 1000 == 0:
                print(
                    f"Iter-{it};"
                    f" D_loss: {D1_loss.item() + D2_loss.item():.4};"
                    f" G_loss: {G_loss.item():.4}"
                )

                real1: torch.Tensor = X1.data[:num_samples]
                real2: torch.Tensor = X2.data[:num_samples]
                samples1: torch.Tensor = X1_sample.data[:num_samples]
                samples2: torch.Tensor = X2_sample.data[:num_samples]
                real1 = real1.view(-1, 1, 28, 28)
                real2 = real2.view(-1, 1, 28, 28)
                samples1 = samples1.view(-1, 1, 28, 28)
                samples2 = samples2.view(-1, 1, 28, 28)
                real_generate_samples = torch.cat([real2, samples1, real1, samples2])

                grid = torchvision.utils.make_grid(real_generate_samples, nrow=4)

                writer.image(f"Samples", data=grid, step=it, cmap="Greys_r")

                cnt += 1


if __name__ == "__main__":
    main()
