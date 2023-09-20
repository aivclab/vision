#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
           """

import time
import torch
import torch.nn
import torchvision
from draugr.torch_utilities import TensorBoardPytorchWriter
from itertools import chain
from neodroidvision import PROJECT_APP_PATH
from torch import optim
from torch.autograd import Variable

from .gan_utilities import reset_grads, sample_x

BATCH_SIZE = 32


def main():
    """description"""
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
    x_dim = 28 * 28
    h_dim = 128
    cnt = 0
    lr = 1e-4
    n_critics = 3
    lam1, lam2 = 100, 100
    num_samples = 4

    generator1 = torch.nn.Sequential(
        torch.nn.Linear(x_dim + z_dim, h_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(h_dim, x_dim),
        torch.nn.Sigmoid(),
    )

    generator2 = torch.nn.Sequential(
        torch.nn.Linear(x_dim + z_dim, h_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(h_dim, x_dim),
        torch.nn.Sigmoid(),
    )

    discriminator1 = torch.nn.Sequential(
        torch.nn.Linear(x_dim, h_dim), torch.nn.ReLU(), torch.nn.Linear(h_dim, 1)
    )

    discriminator2 = torch.nn.Sequential(
        torch.nn.Linear(x_dim, h_dim), torch.nn.ReLU(), torch.nn.Linear(h_dim, 1)
    )

    generators_solver = optim.RMSprop(
        chain(generator1.parameters(), generator2.parameters()), lr=lr
    )
    discriminator1_solver = optim.RMSprop(discriminator1.parameters(), lr=lr)
    discriminator2_solver = optim.RMSprop(discriminator2.parameters(), lr=lr)

    x_train = next(iter(mnist))[0]
    half = int(x_train.shape[0] / 2)

    x_train1 = x_train[:half]

    x_train2: torch.Tensor = x_train[half:]
    x_train2 = x_train2.rot90(dims=(2, 3))

    x_train1 = x_train1.reshape(-1, 28 * 28)
    x_train2 = x_train2.reshape(-1, 28 * 28)

    del x_train  # Cleanup

    with TensorBoardPytorchWriter(
        PROJECT_APP_PATH.user_log / str(time.time())
    ) as writer:
        for it in range(1000000):
            for _ in range(n_critics):
                # Sample data
                z1 = Variable(torch.randn(BATCH_SIZE, z_dim))
                z2 = Variable(torch.randn(BATCH_SIZE, z_dim))
                x1 = sample_x(x_train1, BATCH_SIZE)
                x2 = sample_x(x_train2, BATCH_SIZE)

                # D1
                x2_sample = generator1(torch.cat([x1, z1], -1))  # G1: X1 -> X2
                d1_real = discriminator1(x2)
                d1_fake = discriminator1(x2_sample)

                d1_loss = -(torch.mean(d1_real) - torch.mean(d1_fake))

                d1_loss.backward(retain_graph=True)
                discriminator1_solver.step()

                for p in discriminator1.parameters():  # Weight clipping
                    p.data.clamp_(-0.01, 0.01)

                reset_grads([generator1, generator2, discriminator1, discriminator2])

                # D2
                x1_sample = generator2(torch.cat([x2, z2], -1))  # G2: X2 -> X1
                d2_real = discriminator2(x1)
                d2_fake = discriminator2(x1_sample)

                d2_loss = -(torch.mean(d2_real) - torch.mean(d2_fake))

                d2_loss.backward()
                discriminator2_solver.step()

                for p in discriminator2.parameters():  # Weight clipping
                    p.data.clamp_(-0.01, 0.01)

                reset_grads([generator1, generator2, discriminator1, discriminator2])

            # Generator
            z1 = Variable(torch.randn(BATCH_SIZE, z_dim))
            z2 = Variable(torch.randn(BATCH_SIZE, z_dim))
            x1 = sample_x(x_train1, BATCH_SIZE)
            x2 = sample_x(x_train2, BATCH_SIZE)

            x1_sample = generator2(torch.cat([x2, z2], 1))
            x2_sample = generator1(torch.cat([x1, z1], 1))

            x1_recon = generator2(torch.cat([x2_sample, z2], 1))
            x2_recon = generator1(torch.cat([x1_sample, z1], 1))

            d1_fake = discriminator1(x1_sample)
            d2_fake = discriminator2(x2_sample)

            g_loss = -torch.mean(d1_fake) - torch.mean(d2_fake)
            reg1 = lam1 * torch.mean(torch.sum(torch.abs(x1_recon - x1), 1))
            reg2 = lam2 * torch.mean(torch.sum(torch.abs(x2_recon - x2), 1))

            g_loss += reg1 + reg2

            g_loss.backward()
            generators_solver.step()
            reset_grads([generator1, generator2, discriminator1, discriminator2])

            if it % 1000 == 0:
                print(
                    f"Iter-{it};"
                    f" D_loss: {d1_loss.item() + d2_loss.item():.4};"
                    f" g_loss: {g_loss.item():.4}"
                )

                real1: torch.Tensor = x1.data[:num_samples]
                real2: torch.Tensor = x2.data[:num_samples]
                samples1: torch.Tensor = x1_sample.data[:num_samples]
                samples2: torch.Tensor = x2_sample.data[:num_samples]
                real1 = real1.view(-1, 1, 28, 28)
                real2 = real2.view(-1, 1, 28, 28)
                samples1 = samples1.view(-1, 1, 28, 28)
                samples2 = samples2.view(-1, 1, 28, 28)
                real_generate_samples = torch.cat([real2, samples1, real1, samples2])

                grid = torchvision.utils.make_grid(
                    real_generate_samples, nrow=4
                ).unsqueeze(0)

                writer.image(f"Samples", data=grid, step=it, cmap="Greys_r")

                cnt += 1


if __name__ == "__main__":
    main()
