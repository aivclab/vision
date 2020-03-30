import itertools
from functools import reduce

import numpy
from matplotlib import pyplot


def plot_img_array(img_array: numpy.ndarray, n_col: int = 3) -> None:
    """

  :param img_array:
  :type img_array:
  :param n_col:
  :type n_col:
  :return:
  :rtype:
  """
    n_row = len(img_array) // n_col

    f, plots = pyplot.subplots(
        n_row, n_col, sharex="all", sharey="all", figsize=(n_col * 4, n_row * 4)
    )

    for i in range(len(img_array)):
        plots[i // n_col, i % n_col].imshow(img_array[i])


def plot_side_by_side(img_arrays) -> None:
    flatten_list = reduce(lambda x, y: x + y, zip(*img_arrays))

    plot_img_array(numpy.array(flatten_list), n_col=len(img_arrays))


def plot_errors(results_dict, title) -> None:
    markers = itertools.cycle(("+", "x", "o"))

    pyplot.title(f"{title}")

    for label, result in sorted(results_dict.items()):
        pyplot.plot(result, marker=next(markers), label=label)
        pyplot.ylabel("dice_coef")
        pyplot.xlabel("epoch")
        pyplot.legend(loc=3, bbox_to_anchor=(1, 0))

    pyplot.show()


def masks_to_color_img(masks: numpy.ndarray) -> numpy.ndarray:
    height, width, mask_channels = masks.shape
    color_channels = 3
    color_image = numpy.zeros((height, width, color_channels), dtype=numpy.uint8) * 255

    for y in range(height):
        for x in range(width):
            for mc in range(mask_channels):
                color_image[y, x, mc % color_channels] = masks[y, x, mc]

    return color_image.astype(numpy.uint8)


def plot_prediction(img_array, labels, max_pred, pred, n_col: int = 3) -> None:
    n_row = len(img_array) // n_col

    f, plots = pyplot.subplots(
        n_row, n_col, sharex="all", sharey="all", figsize=(n_col * 4, n_row * 4)
    )

    for i in range(len(img_array)):
        plots[i // n_col, i % n_col].imshow(img_array[i])
        plots[i // n_col, i % n_col].set_title(
            f"truth:{labels[i]},\n max_pred:{max_pred[i]},\n pred:{pred[i]}", fontsize=8
        )
