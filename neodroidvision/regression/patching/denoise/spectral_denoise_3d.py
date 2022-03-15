#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/04/2020
           """

import numpy

__all__ = ["fft3_im_denoise"]


def fft3_im_denoise(img: numpy.ndarray, keep_fraction: float = 0.1) -> numpy.ndarray:
    """
    a blur with an FFT

    Implements, via FFT, the following convolution:

    .. math::

    f_1(t) = \int dt'\, K(t-t') f_0(t')

    .. math::

    \tilde{f}_1(\omega) = \tilde{K}(\omega) \tilde{f}_0(\omega)

    # keep_fraction - Define the fraction of coefficients (in each direction) we keep

    Compute the 3d FFT of the input image
    Filter in FFT
    Reconstruct the final image

    :param keep_fraction:
    :type keep_fraction:
    :param img:
    :type img:
    :return:
    :rtype:"""
    assert 0.0 < keep_fraction < 1.0
    im_fft = numpy.fft.fftn(img)

    im_fft_cp = im_fft  # .copy()
    n_r, n_c, n_a = im_fft_cp.shape  # num row, column, aisle

    im_fft_cp[int(n_r * keep_fraction) : int(n_r * (1 - keep_fraction))] = 0
    im_fft_cp[:, int(n_c * keep_fraction) : int(n_c * (1 - keep_fraction))] = 0
    im_fft_cp[:, :, int(n_a * keep_fraction) : int(n_a * (1 - keep_fraction))] = 0

    return numpy.fft.ifftn(im_fft_cp).real
