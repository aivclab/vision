"""
Train a hierarchical VQ-VAE on 256x256 images.
"""

import argparse
import itertools
import os

import numpy as np
import torch
import torch.optim as optim
from PIL import Image

from neodroidvision import PROJECT_APP_PATH
from samples.reconstruction.vq_vae2.hierarchical.data import load_images
from samples.reconstruction.vq_vae2.hierarchical.model import make_vae

VAE_PATH = PROJECT_APP_PATH.user_data / 'vae.pt'


def main():
  args = arg_parser().parse_args()
  device = torch.device(args.device)
  model = make_vae()
  if os.path.exists(VAE_PATH):
    model.load_state_dict(torch.load(VAE_PATH, map_location='cpu'))
  model.to(device)
  optimizer = optim.Adam(model.parameters())
  data = load_images(args.data)
  for i in itertools.count():
    images = next(data).to(device)
    terms = model(images)
    print('step %d: mse=%f mse_top=%f' %
          (i, terms['losses'][-1].item(), terms['losses'][0].item()))
    optimizer.zero_grad()
    terms['loss'].backward()
    optimizer.step()
    model.revive_dead_entries()
    if not i % 30:
      torch.save(model.state_dict(), VAE_PATH)
      save_reconstructions(model, images)


def save_reconstructions(vae, images):
  vae.eval()
  with torch.no_grad():
    recons = [torch.clamp(x, 0, 1).permute(0, 2, 3, 1).detach().cpu().numpy()
              for x in vae.full_reconstructions(images)]
  vae.train()
  top_recons, real_recons = recons
  images = images.permute(0, 2, 3, 1).detach().cpu().numpy()

  columns = np.concatenate([top_recons, real_recons, images], axis=-2)
  columns = np.concatenate(columns, axis=0)
  Image.fromarray((columns * 255).astype('uint8')).save(PROJECT_APP_PATH.user_data / 'reconstructions.png')


def arg_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('data', help='data directory')
  parser.add_argument('--device', help='torch device', default='cuda')
  return parser


if __name__ == '__main__':
  main()
