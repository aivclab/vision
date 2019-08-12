"""
Produce reconstructions from the VQ-VAE.
"""

import argparse
import os
import sys

import torch

from neodroidvision import PROJECT_APP_PATH
from samples.reconstruction.vq_vae2.text.data import load_text_samples
from samples.reconstruction.vq_vae2.text.model import make_vae

VAE_PATH = PROJECT_APP_PATH.user_data / 'vae.pt'


def main():
  args = arg_parser().parse_args()
  device = torch.device(args.device)

  vae = make_vae()
  if os.path.exists(VAE_PATH):
    vae.load_state_dict(torch.load(VAE_PATH, map_location='cpu'))
  vae.to(device)

  batch = next(load_text_samples(args.data, 1, args.context_len))
  batch = batch.to(device)
  print_bytes('Original', batch[0])
  recons = vae.full_reconstructions(batch)
  for j, recon in enumerate(recons):
    print_bytes('Recon %d' % j, torch.argmax(recon[0], dim=-1))


def print_bytes(headline, inputs):
  buf = bytes(inputs.detach().cpu().numpy().tolist())
  sys.stdout.buffer.write(bytes(headline, 'utf-8'))
  sys.stdout.buffer.write(b': ')
  for i, single_byte in enumerate(buf):
    b = bytes([single_byte])
    if b == b'\n':
      sys.stdout.buffer.write(b'\\n')
    elif b == b'\r':
      sys.stdout.buffer.write(b'\\r')
    else:
      sys.stdout.buffer.write(b)
  sys.stdout.buffer.write(b'\n')
  sys.stdout.buffer.flush()


def arg_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--device', help='torch device', default='cuda')
  parser.add_argument('--context-len', help='context size in bytes', default=512, type=int)
  parser.add_argument('data', help='data file')
  return parser


if __name__ == '__main__':
  main()
