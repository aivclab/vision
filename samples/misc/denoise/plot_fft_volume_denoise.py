from pathlib import Path

import numpy
from draugr.numpy_utilities.raster_grid_3d_masking import sphere_mask
from matplotlib import pyplot

from neodroidvision.regression.denoise.spectral_denoise_3d import fft3_im_denoise

if __name__ == "__main__":


  def main(im_raw):
    """

    :param im_raw:
    :type im_raw:"""
    #pyplot.figure()
    #pyplot.imshow(im_raw, pyplot.cm.gray)
    #pyplot.title("Original image")
    index = (*(numpy.array(im_raw.shape) // 2),)
    print(index)
    print(im_raw[index])
    im_denoised = fft3_im_denoise(im_raw, keep_fraction=0.2)
    print(im_denoised[index])
    #pyplot.figure()
    #pyplot.imshow(im_denoised, pyplot.cm.gray)
    #pyplot.title("Reconstructed Image")

  numpy.random.seed(42)
  im22 = numpy.random.random((100,100,100))
  im22 = sphere_mask(*im22.shape).astype(float)/2 * im22
  im22 = sphere_mask(*im22.shape, radius=5).astype(float) * im22

  main(im22)

  pyplot.show()
