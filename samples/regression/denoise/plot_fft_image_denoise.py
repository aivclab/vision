from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neodroidvision.regression.denoise.spectral_denoise import fft_im_denoise

if __name__ == "__main__":

    im22 = plt.imread(
        str(Path.home() / "Data" / "Datasets" / "Denoise" / "moonlanding.png")
    ).astype(float)

    def plot_spectrum(im_fft):
        """

    :param im_fft:
    :type im_fft:
    """
        from matplotlib.colors import LogNorm

        # A logarithmic colormap
        plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
        plt.colorbar()

    def main(im_raw):
        """

    :param im_raw:
    :type im_raw:
    """
        plt.figure()
        plt.imshow(im_raw, plt.cm.gray)
        plt.title("Original image")

        im_denoised = fft_im_denoise(im_raw)

        plt.figure()
        plt.imshow(im_denoised, plt.cm.gray)
        plt.title("Reconstructed Image")

    def blur_im(im):
        """

    :param im:
    :type im:
    """
        ############################################################
        # Easier and better: :func:`scipy.ndimage.gaussian_filter`
        ############################################################
        #
        # Implementing filtering directly with FFTs is tricky and time consuming.
        # We can use the Gaussian filter from :mod:`scipy.ndimage`

        from scipy import ndimage

        im_blur = ndimage.gaussian_filter(im, 4)

        plt.figure()
        plt.imshow(im_blur, plt.cm.gray)
        plt.title("Blurred image")

    main(im22)

    plt.show()
