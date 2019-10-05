#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
import torch
from PIL import Image
from imageio import imwrite

__author__ = 'Christian Heider Nielsen'
__doc__ = ''


def compile_encoding_image(images, size, resize_factor=1.0):
  h, w = images.shape[1], images.shape[2]

  h_ = int(h * resize_factor)
  w_ = int(w * resize_factor)

  img = numpy.zeros((h_ * size[0], w_ * size[1]))

  for idx, image in enumerate(images):
    i = int(idx % size[1])
    j = int(idx / size[1])

    image_ = numpy.array(Image.fromarray(image).resize((w_, h_), resampe=Image.BICUBIC))

    img[j * h_:j * h_ + h_, i * w_:i * w_ + w_] = image_

  return img


def sample_2d_latent_vectors(encoding_space,
                             n_img_x,
                             n_img_y):
  vectors = numpy.rollaxis(numpy.mgrid[encoding_space:-encoding_space:n_img_y * 1j,
                           encoding_space:-encoding_space:n_img_x * 1j],
                           0,
                           3)

  return torch.FloatTensor([vectors.reshape([-1, 2])])


def plot_manifold(model,
                  out_path,
                  n_img_x=20,
                  n_img_y=20,
                  img_h=28,
                  img_w=28,
                  sample_range=1):
  vectors = sample_2d_latent_vectors(sample_range, n_img_x, n_img_y).to('cuda')
  encodings = model._decoder(vectors).to('cpu')
  images = encodings.reshape(n_img_x * n_img_y, img_h, img_w)
  imwrite(out_path, compile_encoding_image(images, [n_img_y, n_img_x]))
