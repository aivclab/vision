"""
Train a PixelCNN on MNIST using a pre-trained VQ-VAE.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim

from neodroidvision import PROJECT_APP_PATH
from neodroidvision.reconstruction.vqvae2.models import Generator, make_vq_vae
from samples.reconstruction.vq_vae2.mnist import load_images

LATENT_SIZE = 16
LATENT_COUNT = 32
BATCH_SIZE = 32
LR = 1e-3
DEVICE = torch.device('cuda')
VAE_PATH = PROJECT_APP_PATH.user_data / 'vae.pt'
GEN_PATH = PROJECT_APP_PATH.user_data / 'gen.pt'


def main():
  vae = make_vq_vae(LATENT_SIZE, LATENT_COUNT)
  vae.load_state_dict(torch.load(VAE_PATH, map_location='cpu'))
  vae.to(DEVICE)
  vae.eval()

  generator = Generator(LATENT_COUNT)
  if os.path.exists(GEN_PATH):
    generator.load_state_dict(torch.load(GEN_PATH, map_location='cpu'))
  generator.to(DEVICE)

  optimizer = optim.Adam(generator.parameters(), lr=LR)
  loss_fn = nn.CrossEntropyLoss()

  test_images = load_images(train=False)
  for batch_idx, images in enumerate(load_images()):
    images = images.to(DEVICE)
    losses = []
    for img_set in [images, next(test_images).to(DEVICE)]:
      _, _, encoded = vae.encoders[0](img_set)
      logits = generator(encoded)
      logits = logits.permute(0, 2, 3, 1).contiguous()
      logits = logits.view(-1, logits.shape[-1])
      losses.append(loss_fn(logits, encoded.view(-1)))
    optimizer.zero_grad()
    losses[0].backward()
    optimizer.step()
    print('train=%f test=%f' % (losses[0].item(), losses[1].item()))
    if not batch_idx % 100:
      torch.save(generator.state_dict(), GEN_PATH)


if __name__ == '__main__':
  main()
