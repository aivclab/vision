#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional, Tuple

import numpy
import torch
from PIL import Image
from draugr.torch_utilities import global_torch_device
from torch.nn.functional import one_hot

__author__ = "Christian Heider Nielsen"
__doc__ = r""""""

from numpy import ndarray

from warg import Number


def compile_encoding_image(
    images: ndarray, size: Tuple, resize_factor: Number = 1.0
) -> ndarray:
    """


    :param images:
    :param size:
    :param resize_factor:
    :return:"""
    h, w = images.shape[1], images.shape[2]

    h_ = int(h * resize_factor)
    w_ = int(w * resize_factor)
    r = []
    if len(images.shape) > 3:
        r = images.shape[3:]

    img = numpy.zeros((h_ * size[0], w_ * size[1], *r))

    for idx, image in enumerate(images):
        i = int(idx % size[1])
        j = int(idx / size[1])

        image_ = numpy.array(
            Image.fromarray(image).resize((w_, h_), resample=Image.BICUBIC)
        )

        img[j * h_ : j * h_ + h_, i * w_ : i * w_ + w_] = image_

    return img


def sample_2d_latent_vectors(
    encoding_space: Number, n_img_x: int, n_img_y: int
) -> torch.FloatTensor:
    """

    :param encoding_space:
    :param n_img_x:
    :param n_img_y:
    :return:"""

    return torch.FloatTensor(
        [
            numpy.rollaxis(
                numpy.mgrid[
                    encoding_space : -encoding_space : n_img_y * 1j,
                    encoding_space : -encoding_space : n_img_x * 1j,
                ],
                0,
                3,
            ).reshape([-1, 2])
        ]
    )


def plot_conditioned_manifold(
    model: torch.nn.Module,
    condition: torch.Tensor,
    *,
    out_path: Path = None,
    n_img_x: int = 20,
    n_img_y: int = 20,
    img_h: int = 28,
    img_w: int = 28,
    sample_range: Number = 1,
    device: Optional[torch.device] = global_torch_device()
) -> None:
    condition_vector = torch.arange(0, 10, device=device).long().unsqueeze(1)
    sample = model.sample(
        one_hot(condition_vector, 10).to(device=device),
        num=condition_vector.size(0),
    )
    # TODO: FINISH


def plot_manifold(
    model: torch.nn.Module,
    *,
    out_path: Path = None,
    n_img_x: int = 20,
    n_img_y: int = 20,
    img_h: int = 28,
    img_w: int = 28,
    sample_range: Number = 1,
    device: Optional[torch.device] = global_torch_device()
) -> None:
    """

    :param model:
    :param out_path:
    :param n_img_x:
    :param n_img_y:
    :param img_h:
    :param img_w:
    :param sample_range:
    :return:"""
    vectors = sample_2d_latent_vectors(sample_range, n_img_x, n_img_y).to(device)
    encodings = torch.sigmoid(model(vectors)).to("cpu")
    images = encodings.reshape(n_img_x * n_img_y, img_h, img_w, -1).numpy()
    images *= 255
    images = numpy.uint8(images)
    compiled = compile_encoding_image(images, (n_img_y, n_img_x))
    if out_path:
        from imageio import imwrite

        imwrite(str(out_path), compiled)
    return compiled
