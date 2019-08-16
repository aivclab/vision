#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from math import inf
from pathlib import Path

import torch
import torch.utils.data
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from draugr.writers import TensorBoardPytorchWriter, Writer
from neodroidvision import PROJECT_APP_PATH
from neodroidvision.data.vgg_face2 import VggFaces2
from neodroidvision.reconstruction.vae.architectures.beta_vae import HigginsVae, VAE
from neodroidvision.reconstruction.visualisation.encoder_utilities import plot_manifold
from objectives import kl_divergence, reconstruction_loss

__author__ = 'cnheider'
__doc__ = r''' 
  Training for BetaVae's
'''

torch.manual_seed(82375329)
LOWEST_L = inf
import multiprocessing

core_count= min(8,multiprocessing.cpu_count()-1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DL_KWARGS = {'num_workers':core_count,
             'pin_memory':True} if torch.cuda.is_available() else {}
BASE_PATH = (PROJECT_APP_PATH.user_data / 'bvae')
if not BASE_PATH.exists():
  BASE_PATH.mkdir(parents=True)

INPUT_SIZE = 64
CHANNELS = 3

BATCH_SIZE = 1024
EPOCHS = 1000
LR = 3e-3
ENCODING_SIZE = 10
DATASET = VggFaces2(Path(f'/home/heider/Data/vggface2'),
                    split='train',
                    resize_s=INPUT_SIZE)
MODEL: VAE = HigginsVae(CHANNELS,
                        latent_size=ENCODING_SIZE).to(DEVICE)
BETA = 4


def loss_function(reconstruction, original, mean, log_var, beta=1):
  return reconstruction_loss(reconstruction, original) + beta * kl_divergence(mean, log_var)


def train_model(model,
                optimiser,
                epoch_i: int,
                metric_writer: Writer,
                loader: DataLoader,
                log_interval=10):
  model.train()
  train_accum_loss = 0
  generator = tqdm(enumerate(loader))
  for batch_idx, (original, *_) in generator:
    original = original.to(DEVICE)

    optimiser.zero_grad()
    reconstruction, mean, log_var = model(original)
    loss = loss_function(reconstruction, original, mean, log_var)
    loss.backward()
    optimiser.step()

    train_accum_loss += loss.item()
    metric_writer.scalar('train_loss', loss.item())

    if batch_idx % log_interval == 0:
      generator.set_description(f'Train Epoch: {epoch_i}'
                                f' [{batch_idx * len(original)}/'
                                f'{len(loader.dataset)}'
                                f' ({100. * batch_idx / len(loader):.0f}%)]\t'
                                f'Loss: {loss.item() / len(original):.6f}')
    break
  print(f'====> Epoch: {epoch_i}'
        f' Average loss: {train_accum_loss / len(loader.dataset):.4f}')


def run_model(model: VAE,
              epoch_i: int,
              metric_writer: Writer,
              loader: DataLoader,
              save_images: bool = True):
  global LOWEST_L
  model.eval()
  test_accum_loss = 0

  with torch.no_grad():
    for i, (original, labels, *_) in enumerate(loader):
      original = original.to(DEVICE)

      reconstruction, mean, log_var = model(original)
      loss = loss_function(reconstruction,
                           original,
                           mean,
                           log_var).item()

      test_accum_loss += loss
      metric_writer.scalar('test_loss', test_accum_loss)

      if save_images:
        if i == 0:
          n = min(original.size(0), 8)
          comparison = torch.cat([original[:n],
                                  reconstruction[:n]])
          save_image(comparison.cpu(),  # Torch save images
                     str(BASE_PATH / f'reconstruction_{str(epoch_i)}.png'), nrow=n)
          '''
          scatter_plot_encoding_space(str(BASE_PATH /
                                          f'encoding_space_{str(epoch_i)}.png'),
                                      mean.to('cpu').numpy(),
                                      log_var.to('cpu').numpy(),
                                      labels)
          '''
      break

  # test_loss /= len(loader.dataset)
  test_accum_loss /= loader.batch_size
  print(f'====> Test set loss: {test_accum_loss:.4f}')
  torch.save(model.state_dict(), BASE_PATH / f'model_state_dict{str(epoch_i)}.pth')

  if LOWEST_L > test_accum_loss:
    LOWEST_L = test_accum_loss
    torch.save(model.state_dict(), BASE_PATH / f'best_state_dict.pth')


if __name__ == "__main__":

  def main():

    '''
    ds = [datasets.MNIST(PROJECT_APP_PATH.user_data,
                         train=True,
                         download=True,
                         transform=transforms.ToTensor()), datasets.MNIST(PROJECT_APP_PATH.user_data,
                                                                          train=False,
                                                                          transform=transforms.ToTensor())]
                                                                          '''

    dataset_loader = DataLoader(DATASET,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                **DL_KWARGS)

    optimiser = optim.Adam(MODEL.parameters(),
                           lr=LR,
                           betas=(0.9, 0.999))

    with TensorBoardPytorchWriter(PROJECT_APP_PATH.user_log / 'VggFace2'
                                  / 'BetaVAE' /
                                  f'{time.time()}') as metric_writer:
      for epoch in range(1, EPOCHS + 1):
        train_model(MODEL, optimiser, epoch, metric_writer, dataset_loader)
        run_model(MODEL, epoch, metric_writer, dataset_loader)
        with torch.no_grad():
          a = MODEL.generate(1, device=DEVICE).view(CHANNELS, INPUT_SIZE, INPUT_SIZE)
          A = DATASET.inverse_transform(a)
          A.save(str(BASE_PATH / f"sample_{str(epoch)}.png"))
          if ENCODING_SIZE == 2:
            plot_manifold(MODEL,
                          out_path=str(BASE_PATH /
                                       f"manifold_{str(epoch)}.png"),
                          img_w=INPUT_SIZE,
                          img_h=INPUT_SIZE)


  main()
