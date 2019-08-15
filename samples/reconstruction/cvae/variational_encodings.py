#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from math import inf
from pathlib import Path

import time
from tqdm import tqdm
from vgg_face2 import VggFaces2

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


from torchvision.utils import save_image
from draugr.writers import Writer,TensorBoardPytorchWriter

lowest_l = inf
ENCODING_SIZE = 2


def train_model(epoch_i, metric_writer: Writer, loader):
  model.train()
  train_loss = 0
  generator = tqdm(enumerate(loader))
  for batch_idx, (data, *_) in generator:

    data = data.to(DEVICE)
    optimizer.zero_grad()
    recon_batch, mean, log_var = model(data)
    loss = FlatNormalVAE.loss_function(recon_batch, data, mean, log_var)
    loss.backward()
    train_loss += loss.item()
    optimizer.step()
    metric_writer.scalar('train_loss', loss.item())

    if batch_idx % args.log_interval == 0:
      generator.set_description(f'Train Epoch: {epoch_i}'
                                f' [{batch_idx * len(data)}/{len(loader.dataset)}'
                                f' ({100. * batch_idx / len(loader):.0f}%)]\t'
                                f'Loss: {loss.item() / len(data):.6f}')

  print(f'====> Epoch: {epoch_i}'
        f' Average loss: {train_loss / len(loader.dataset):.4f}')


def run_model(epoch_i, metric_writer, loader):
  global lowest_l
  model.eval()
  test_loss = 0

  with torch.no_grad():
    for i, (data, labels, *_) in enumerate(loader):
      data = data.to(DEVICE)
      recon_batch, mean, log_var = model(data)
      test_loss += FlatNormalVAE.loss_function(recon_batch,
                                               data,
                                               mean,
                                               log_var).item()
      metric_writer.scalar('test_loss', test_loss)
      if i == 0:
        n = min(data.size(0), 8)
        comparison = torch.cat([data[:n],
                                recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
        save_image(comparison.cpu(),
                   str(result_base_path / f'reconstruction_{str(epoch_i)}.png'), nrow=n)

        scatter_plot_encoding_space(str(result_base_path /
                                        f'encoding_space_{str(epoch_i)}.png'),
                                    mean.to('cpu').numpy(),
                                    log_var.to('cpu').numpy(),
                                    labels)

  test_loss /= len(loader.dataset)
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

  kwargs = {'num_workers':4, 'pin_memory':True} if args.cuda else {}

  '''
  ds = [datasets.MNIST(PROJECT_APP_PATH.user_data,
                       train=True,
                       download=True,
                       transform=transforms.ToTensor()), datasets.MNIST(PROJECT_APP_PATH.user_data,
                                                                        train=False,
                                                                        transform=transforms.ToTensor())]
                                                                        '''
  dset = 'test'

  ds = VggFaces2(Path('/home/heider/Data/vggface2/{dset}'),
                 Path('/home/heider/Data/vggface2/{dset}_list.txt'),
                 Path('/home/heider/Data/vggface2/identity_meta.csv'),
                 split=dset)

  train_loader = torch.utils.data.DataLoader(ds,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             **kwargs)

  test_loader = torch.utils.data.DataLoader(ds,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            **kwargs)

  result_base_path = (PROJECT_APP_PATH.user_data / 'results')

  if not result_base_path.exists():
    result_base_path.mkdir(parents=True)

  model = FlatNormalVAE(encoding_size=ENCODING_SIZE).to(DEVICE)
  optimizer = optim.Adam(model.parameters(), lr=1e-3)

  with TensorBoardPytorchWriter(PROJECT_APP_PATH.user_log/f'{time.time()}') as metric_writer:
    for epoch in range(1, args.epochs + 1):
      train_model(epoch, metric_writer, train_loader)
      run_model(epoch, metric_writer, test_loader)
      with torch.no_grad():
        save_image(model.sample(device=DEVICE).view(64, 1, 28, 28),
                   str(result_base_path / f"sample_{str(epoch)}.png"))
        if ENCODING_SIZE == 2:
          plot_manifold(model,
                        out_path=str(result_base_path /
                                     f"manifold_{str(epoch)}.png"))
