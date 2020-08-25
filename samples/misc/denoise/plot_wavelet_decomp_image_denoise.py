from pathlib import Path

import numpy
import pywt
from matplotlib import pyplot

from neodroidvision.regression.denoise.spectral_denoise import fft_im_denoise

if __name__ == "__main__":

    def wavelet_denoise(im):
        """

:param im:
:type im:
"""
        mother_wavelet = "db1"  # Daubechies wavelet 1
        levels = 4
        keep = 1 / 1e2  # percent

        coef = pywt.wavedec2(im, wavelet=mother_wavelet, level=levels)

        coef_array, coef_slices = pywt.coeffs_to_array(coef)

        Csort = numpy.sort(numpy.abs(coef_array.reshape(-1)))

        coef_filt = pywt.array_to_coeffs(
            coef_array
            * (
                numpy.abs(coef_array) > Csort[int(numpy.floor((1 - keep) * len(Csort)))]
            ),
            coef_slices,
            output_format="wavedec2",
        )

        recon = pywt.waverec2(coef_filt, wavelet=mother_wavelet)

        return recon

    def main(im_raw):
        """

:param im_raw:
:type im_raw:
"""
        pyplot.figure()
        pyplot.imshow(im_raw, pyplot.cm.gray)
        pyplot.title("Original image")

        im_denoised = wavelet_denoise(im_raw)

        pyplot.figure()
        pyplot.imshow(im_denoised, pyplot.cm.gray)
        pyplot.title("Reconstructed Image")

    im22 = (
        pyplot.imread(
            "/home/heider/Pictures/Christian Heider Nielsen_scaled.png"
            # str(Path.home() / "Data" / "Datasets" / "Denoise" / "moonlanding.png")
        )
        .astype(float)
        .mean(-1)
    )
    main(im22)

    pyplot.show()
