#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from math import inf
from pathlib import Path

from torch.nn.functional import binary_cross_entropy, mse_loss
from torch.utils.data import DataLoader
from tqdm import tqdm

from neodroidvision import PROJECT_APP_PATH
from neodroidvision.data.vgg_face2 import VggFaces2
from neodroidvision.reconstruction.vae.architectures.beta_vae import HigginsBetaVae, VAE, BurgessBetaVae
from neodroidvision.reconstruction.visualisation.encoder_utilities import plot_manifold
from neodroidvision.reconstruction.visualisation.encoding_space import scatter_plot_encoding_space

__author__ = 'cnheider'
__doc__ = ''

import torch
import torch.utils.data
from torch import optim

from torchvision.utils import save_image
from draugr.writers import Writer, TensorBoardPytorchWriter

torch.manual_seed(82375329)
LOWEST_L = inf
ENCODING_SIZE = 10
INPUT_SIZE = 64
CHANNELS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DL_KWARGS = {'num_workers':4, 'pin_memory':True} if torch.cuda.is_available() else {}
BASE_PATH = (PROJECT_APP_PATH.user_data / 'bvae')
if not BASE_PATH.exists():
  BASE_PATH.mkdir(parents=True)
BATCH_SIZE = 256
EPOCHS = 1000
LR = 3e-3
DATASET = VggFaces2(Path(f'/home/heider/Data/vggface2'),
                    split='train',
                    resize_s=INPUT_SIZE)
BETA=4

def reconstruction_loss(reconstruction,original):
    batch_size = original.size(0)
    assert batch_size != 0

      #recon_loss = F.binary_cross_entropy_with_logits(reconstruction,
    # original,
    # size_average=False).div(batch_size)

    #reconstruction = torch.sigmoid(reconstruction)
    recon_loss = mse_loss(reconstruction, original, size_average=False).div(batch_size)

    return recon_loss


def kl_divergence(mean, log_var):
  batch_size = mean.size(0)
  assert batch_size != 0
  if mean.data.ndimension() == 4:
    mean = mean.view(mean.size(0), mean.size(1))
  if log_var.data.ndimension() == 4:
    log_var = log_var.view(log_var.size(0), log_var.size(1))

  klds = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
  total_kld = klds.sum(1).mean(0, True)
  dimension_wise_kld = klds.mean(0)
  mean_kld = klds.mean(1).mean(0, True)

  return total_kld, dimension_wise_kld, mean_kld


def loss_function(reconstruction, original, mean, log_var, beta=1):
  return reconstruction_loss(reconstruction,original) + beta * kl_divergence(mean,log_var)[0]


def train_model(model,
                optimiser,
                epoch_i: int,
                metric_writer: Writer,
                loader: DataLoader,
                log_interval=10):
  model.train()
  train_loss = 0
  generator = tqdm(enumerate(loader))
  for batch_idx, (original, *_) in generator:

    original = original.to(DEVICE)
    optimiser.zero_grad()
    reconstruction, mean, log_var = model(original)
    loss = loss_function(reconstruction, original, mean, log_var)
    loss.backward()
    train_loss += loss.item()
    optimiser.step()
    metric_writer.scalar('train_loss', loss.item())

    if batch_idx % log_interval == 0:
      generator.set_description(f'Train Epoch: {epoch_i}'
                                f' [{batch_idx * len(original)}/{len(loader.dataset)}'
                                f' ({100. * batch_idx / len(loader):.0f}%)]\t'
                                f'Loss: {loss.item() / len(original):.6f}')

  print(f'====> Epoch: {epoch_i}'
        f' Average loss: {train_loss / len(loader.dataset):.4f}')


def run_model(model: VAE,
              epoch_i: int,
              metric_writer: Writer,
              loader: DataLoader,
              save_images: bool = True):
  global LOWEST_L
  model.eval()
  test_loss = 0

  with torch.no_grad():
    for i, (original, labels, *_) in enumerate(loader):
      original = original.to(DEVICE)
      reconstruction, mean, log_var = model(original)
      test_loss += loss_function(reconstruction,
                                 original,
                                 mean,
                                 log_var).item()
      metric_writer.scalar('test_loss', test_loss)
      if save_images:
        if i == 0:
          n = min(original.size(0), 8)
          comparison = torch.cat([original[:n],
                                  reconstruction[:n]])
          save_image(comparison.cpu(),
                     str(BASE_PATH / f'reconstruction_{str(epoch_i)}.png'), nrow=n)

          scatter_plot_encoding_space(str(BASE_PATH /
                                          f'encoding_space_{str(epoch_i)}.png'),
                                      mean.to('cpu').numpy(),
                                      log_var.to('cpu').numpy(),
                                      labels)
      break

  # test_loss /= len(loader.dataset)
  test_loss /= loader.batch_size
  print('====> Test set loss: {:.4f}'.format(test_loss))
  torch.save(model.state_dict(), BASE_PATH / f'model_state_dict{str(epoch_i)}')

  if LOWEST_L > test_loss:
    LOWEST_L = test_loss
    torch.save(model.state_dict(), BASE_PATH / f'best_state_dict')


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

    model: VAE = BurgessBetaVae(CHANNELS,
                                latent_size=ENCODING_SIZE).to(DEVICE)
    optimiser = optim.Adam(model.parameters(),
                           lr=LR,
                           betas=(0.9, 0.999))

    with TensorBoardPytorchWriter(PROJECT_APP_PATH.user_log / 'VggFace2'
                                  / 'BetaVAE' /
                                  f'{time.time()}') as metric_writer:
      for epoch in range(1, EPOCHS + 1):
        train_model(model, optimiser, epoch, metric_writer, dataset_loader)
        run_model(model, epoch, metric_writer, dataset_loader)
        with torch.no_grad():
          a = model.generate(1, device=DEVICE).view(CHANNELS, INPUT_SIZE, INPUT_SIZE)
          A = DATASET.inverse_transform(a)
          A.save(str(BASE_PATH / f"sample_{str(epoch)}.png"))
          if ENCODING_SIZE == 2:
            plot_manifold(model,
                          out_path=str(BASE_PATH /
                                       f"manifold_{str(epoch)}.png"),
                          img_w=INPUT_SIZE,
                          img_h=INPUT_SIZE)


  main()
