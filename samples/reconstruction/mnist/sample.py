"""
Sample an image from a PixelCNN.
"""

import random

import numpy as np
import torch
from PIL import Image
from vision import PROJECT_APP_PATH

from samples.reconstruction.mnist.model import Generator, make_vq_vae

DEVICE = torch.device('cpu')
VAE_PATH = PROJECT_APP_PATH.user_data / 'vae.pt'
GEN_PATH = PROJECT_APP_PATH.user_data / 'gen.pt'


def main():
  vae = make_vq_vae()
  vae.load_state_dict(torch.load(VAE_PATH, map_location='cpu'))
  vae.to(DEVICE)
  vae.eval()
  generator = Generator()
  generator.load_state_dict(torch.load(GEN_PATH, map_location='cpu'))
  generator.to(DEVICE)

  inputs = np.zeros([4, 7, 7], dtype=np.long)
  for row in range(7):
    for col in range(7):
      with torch.no_grad():
        outputs = torch.softmax(generator(torch.from_numpy(inputs).to(DEVICE)), dim=1)
        for i, out in enumerate(outputs.cpu().numpy()):
          probs = out[:, row, col]
          inputs[i, row, col] = sample_softmax(probs)
    print('done row', row)
  embedded = vae.encoders[0].vq.embed(torch.from_numpy(inputs).to(DEVICE))
  decoded = torch.clamp(vae.decoders[0]([embedded]), 0, 1).detach().cpu().numpy()
  decoded = np.concatenate(decoded, axis=1)
  Image.fromarray((decoded * 255).astype(np.uint8)[0]).save(PROJECT_APP_PATH.user_data / 'samples.png')


def sample_softmax(probs):
  number = random.random()
  for i, x in enumerate(probs):
    number -= x
    if number <= 0:
      return i
  return len(probs) - 1


if __name__ == '__main__':
  main()
