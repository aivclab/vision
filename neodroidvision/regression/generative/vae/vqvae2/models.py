"""
A basic PixelCNN + VQ-VAE model.
"""

import torch.nn as nn

from neodroidvision.reconstruction import (
  PixelCNN,
  PixelConvA,
  PixelConvB,
  QuarterDecoder,
  QuarterEncoder,
  VQVAE,
  )


def make_vq_vae(LATENT_SIZE=16, LATENT_COUNT=32):
  return VQVAE(
    [QuarterEncoder(1, LATENT_SIZE, LATENT_COUNT)], [QuarterDecoder(LATENT_SIZE, 1)]
    )


class Generator(nn.Module):
  def __init__(self, LATENT_COUNT=32):
    super().__init__()
    self.encoder = nn.Embedding(LATENT_COUNT, 64)
    self.model = PixelCNN(
      PixelConvA(64, 64),
      PixelConvB(64, 64, norm=True),
      PixelConvB(64, 64, norm=True),
      PixelConvB(64, 64, norm=True),
      PixelConvB(64, 64, norm=True),
      PixelConvB(64, 64, norm=True),
      PixelConvB(64, 64, norm=True),
      PixelConvB(64, 64, norm=True),
      )
    self.to_logits = nn.Conv2d(64, LATENT_COUNT, 1)

  def forward(self, x):
    x = self.encoder(x)
    x = x.permute(0, 3, 1, 2).contiguous()
    out1, out2 = self.model(x)
    return self.to_logits(out1 + out2)
