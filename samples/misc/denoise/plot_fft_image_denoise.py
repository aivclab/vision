from pathlib import Path

import numpy
from matplotlib import pyplot
from neodroidvision.regression.denoise.spectral_denoise import fft_im_denoise

if __name__ == "__main__":

    def plot_spectrum(im_fft):
        """

        :param im_fft:
        :type im_fft:"""
        from matplotlib.colors import LogNorm

        # A logarithmic colormap
        pyplot.imshow(numpy.abs(im_fft), norm=LogNorm(vmin=5))
        pyplot.colorbar()

    def blur_im(im):
        """

        :param im:
        :type im:"""
        ############################################################
        # Easier and better: :func:`scipy.ndimage.gaussian_filter`
        ############################################################
        #
        # Implementing filtering directly with FFTs is tricky and time consuming.
        # We can use the Gaussian filter from :mod:`scipy.ndimage`

        from scipy import ndimage

        im_blur = ndimage.gaussian_filter(im, 4)

        pyplot.figure()
        pyplot.imshow(im_blur, pyplot.cm.gray)
        pyplot.title("Blurred image")

    def main(im_raw):
        """

        :param im_raw:
        :type im_raw:"""
        pyplot.figure()
        pyplot.imshow(im_raw, pyplot.cm.gray)
        pyplot.title("Original image")

        im_denoised = fft_im_denoise(im_raw)

        pyplot.figure()
        pyplot.imshow(im_denoised, pyplot.cm.gray)
        pyplot.title("Reconstructed Image")

    im22 = pyplot.imread(
        str(Path.home() / "Data" / "Datasets" / "Denoise" / "moonlanding.png")
    ).astype(float)

    main(im22)

    pyplot.show()
