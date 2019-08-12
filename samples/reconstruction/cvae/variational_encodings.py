#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from math import inf

from draugr import TensorBoardXWriter
from tqdm import tqdm

from neodroidvision import PROJECT_APP_PATH
from neodroidvision.reconstruction.cvae.archs.flat import FlatNormalVAE
from neodroidvision.reconstruction.cvae.plotting.encoder_utilities import plot_manifold
from neodroidvision.reconstruction.cvae.plotting.encoding_space import scatter_plot_encoding_space

__author__ = 'cnheider'
__doc__ = ''

import argparse

import torch
import torch.utils.data
from torch import optim

from torchvision import datasets, transforms
from torchvision.utils import save_image


def train(epoch, stat_writer, train_loader):
  model.train()
  train_loss = 0
  s = tqdm(enumerate(train_loader))
  for batch_idx, (data, _) in s:
    data = data.to(DEVICE)
    optimizer.zero_grad()
    recon_batch, mean, log_var = model(data)
    loss = FlatNormalVAE.loss_function(recon_batch, data, mean, log_var)
    loss.backward()
    train_loss += loss.item()
    optimizer.step()
    stat_writer.scalar('train_loss', loss.item())

    if batch_idx % args.log_interval == 0:
      s.set_description(f'Train Epoch: {epoch}'
                        f' [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                        f' ({100. * batch_idx / len(train_loader):.0f}%)]\t'
                        f'Loss: {loss.item() / len(data):.6f}')

  print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')


lowest_l = inf
ENCODING_SIZE = 3


def run_model(epoch, stat_writer, test_loader):
  global lowest_l
  model.eval()
  test_loss = 0

  with torch.no_grad():
    for i, (data, labels) in enumerate(test_loader):
      data = data.to(DEVICE)
      recon_batch, mean, log_var = model(data)
      test_loss += FlatNormalVAE.loss_function(recon_batch,
                                               data,
                                               mean,
                                               log_var).item()
      stat_writer.scalar('test_loss', test_loss)
      if i == 0:
        n = min(data.size(0), 8)
        comparison = torch.cat([data[:n],
                                recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
        save_image(comparison.cpu(),
                   str(result_base_path / f'reconstruction_{str(epoch)}.png'), nrow=n)

        scatter_plot_encoding_space(str(result_base_path / f'encoding_space_{str(epoch)}.png'),
                                    mean.to('cpu').numpy(),
                                    log_var.to('cpu').numpy(),
                                    labels)

  test_loss /= len(test_loader.dataset)
  print('====> Test set loss: {:.4f}'.format(test_loss))

  if lowest_l > test_loss:
    lowest_l = test_loss
    torch.save(model.state_dict(), result_base_path / f'best_state_dict')


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='VAE MNIST Example')
  parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                      help='input batch size for training (default: 128)')
  parser.add_argument('--epochs', type=int, default=100, metavar='N',
                      help='number of epochs to train (default: 10)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='enables CUDA training')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')
  args = parser.parse_args()
  args.cuda = not args.no_cuda and torch.cuda.is_available()

  torch.manual_seed(args.seed)

  DEVICE = torch.device("cuda" if args.cuda else "cpu")

  kwargs = {'num_workers':1, 'pin_memory':True} if args.cuda else {}

  tran_loader = torch.utils.data.DataLoader(datasets.MNIST(PROJECT_APP_PATH.user_data,
                                                           train=True,
                                                           download=True,
                                                           transform=transforms.ToTensor()),
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            **kwargs)

  tet_loader = torch.utils.data.DataLoader(datasets.MNIST(PROJECT_APP_PATH.user_data,
                                                          train=False,
                                                          transform=transforms.ToTensor()),
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           **kwargs)

  result_base_path = (PROJECT_APP_PATH.user_data / 'results')

  if not result_base_path.exists():
    result_base_path.mkdir(parents=True)

  model = FlatNormalVAE(encoding_size=ENCODING_SIZE).to(DEVICE)
  optimizer = optim.Adam(model.parameters(), lr=1e-3)

  with TensorBoardXWriter(result_base_path) as stat_w:
    for epoch in range(1, args.epochs + 1):
      train(epoch, stat_w, tran_loader)
      run_model(epoch, stat_w, tet_loader)
      with torch.no_grad():
        save_image(model.sample(device=DEVICE).view(64, 1, 28, 28),
                   str(result_base_path / f"sample_{str(epoch)}.png"))
        if ENCODING_SIZE == 2:
          plot_manifold(model,
                      out_path=str(result_base_path / f"manifold_{str(epoch)}.png"))
