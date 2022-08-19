#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/04/2020
           """

import numpy
from scipy import fftpack

__all__ = ["fft_im_denoise"]


def fft_im_denoise(img: numpy.ndarray, keep_fraction: float = 0.1) -> numpy.ndarray:
    """
    a blur with an FFT

    Implements, via FFT, the following convolution:

    .. math::

    f_1(t) = \int dt'\, K(t-t') f_0(t')

    .. math::

    \tilde{f}_1(\omega) = \tilde{K}(\omega) \tilde{f}_0(\omega)

    # keep_fraction - Define the fraction of coefficients (in each direction) we keep

    Compute the 2d FFT of the input image
    Filter in FFT
    Reconstruct the final image

    :param keep_fraction:
    :type keep_fraction:
    :param img:
    :type img:
    :return:
    :rtype:"""
    assert 0.0 < keep_fraction < 1.0

    im_fft = fftpack.fft2(img)

    # In the lines following, we'll make a copy of the original spectrum and
    # truncate coefficients.
    # Call ff a copy of the original transform. Numpy arrays have a copy
    # method for this purpose.
    im_fft_cp = im_fft  # .copy()
    num_row, num_columns = im_fft_cp.shape

    # Set to zero all rows with indices between r*keep_fraction and
    # r*(1-keep_fraction):
    im_fft_cp[int(num_row * keep_fraction) : int(num_row * (1 - keep_fraction))] = 0
    im_fft_cp[
        :, int(num_columns * keep_fraction) : int(num_columns * (1 - keep_fraction))
    ] = 0

    # pyplot.figure()
    # plot_spectrum(im_fft)
    # pyplot.title('Fourier transform')

    # pyplot.figure()
    # plot_spectrum(im_fft_cp)
    # pyplot.title('Filtered Spectrum')

    # Reconstruct the denoised image from the filtered spectrum, keep only the
    # real part for display.
    return fftpack.ifft2(im_fft_cp).real  # Inverse / Reconstruction
