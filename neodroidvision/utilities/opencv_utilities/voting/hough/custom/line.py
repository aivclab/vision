import math

import imageio
import numpy

from pathlib import Path
from draugr.opencv_utilities import to_gray
from matplotlib import pyplot
from tqdm import tqdm


def hough_line(img, angle_step=1, lines_are_white=True, value_threshold=5):
    """
    Hough transform for lines
    Input:
    img - 2D binary image with nonzeros representing edges
    angle_step - Spacing between angles to use every n-th angle
                 between -90 and 90 degrees. Default step is 1.
    lines_are_white - boolean indicating whether lines to be detected are white
    value_threshold - Pixel values above or below the value_threshold are edges
    Returns:
    accumulator - 2D array of the hough transform accumulator
    theta - array of angles used in computation, in radians.
    rhos - array of rho values. Max size is 2 times the diagonal
           distance of the input image.
    """
    # Rho and Theta ranges
    thetas = numpy.deg2rad(numpy.arange(-90.0, 90.0, angle_step))
    width, height = img.shape
    diag_len = int(round(math.sqrt(width * width + height * height)))
    rhos = numpy.linspace(-diag_len, diag_len, diag_len * 2)

    # Cache some resuable values
    cos_t = numpy.cos(thetas)
    sin_t = numpy.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = numpy.zeros((2 * diag_len, num_thetas), dtype=numpy.uint8)
    # (row, col) indexes to edges
    are_edges = img > value_threshold if lines_are_white else img < value_threshold
    y_idxs, x_idxs = numpy.nonzero(are_edges)

    # Vote in the hough accumulator
    for i in tqdm(range(len(x_idxs))):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in tqdm(range(num_thetas)):
            # Calculate rho. diag_len is added for a positive index
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos


def show_hough_line(img, accumulator, thetas, rhos, save_path=None):
    pyplot.imshow(
        accumulator,
        aspect="auto",
        cmap="jet",
        extent=[numpy.rad2deg(thetas[-1]), numpy.rad2deg(thetas[0]), rhos[-1], rhos[0]],
    )
    if save_path is not None:
        pyplot.savefig(save_path, bbox_inches="tight")
    pyplot.show()


if __name__ == "__main__":
    imgpath = Path.home() / "OneDrive" / "Billeder" / "2.jpg"
    if imgpath.exists():
        img = imageio.imread(imgpath)
        accumulator, thetas, rhos = hough_line(to_gray(img))
        show_hough_line(img, accumulator, thetas, rhos)
    else:
        print(f"could not find {imgpath}")
