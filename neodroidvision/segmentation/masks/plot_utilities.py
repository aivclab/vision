import itertools
from typing import Dict, Sequence, Tuple

import numpy
from matplotlib import pyplot
from warg import Number

__all__ = [
    "plot_errors",
    "masks_to_color_img",
    "plot_prediction",
    "bounding_box_from_mask",
]


def plot_errors(results_dict: Dict, title: str) -> None:
    """

    Args:
      results_dict:
      title:
    """
    markers = itertools.cycle(("+", "x", "o"))

    pyplot.title(f"{title}")

    for label, result in sorted(results_dict.items()):
        pyplot.plot(result, marker=next(markers), label=label)
        pyplot.ylabel("dice_coef")
        pyplot.xlabel("epoch")
        pyplot.legend(loc=3, bbox_to_anchor=(1, 0))

    pyplot.show()


def masks_to_color_img(masks: numpy.ndarray) -> numpy.ndarray:
    """

    Args:
      masks:

    Returns:

    """
    height, width, mask_channels = masks.shape
    color_channels = 3
    color_image = numpy.zeros((height, width, color_channels), dtype=numpy.uint8) * 255

    for y in range(height):
        for x in range(width):
            for mc in range(mask_channels):
                color_image[y, x, mc % color_channels] = masks[y, x, mc]

    return color_image.astype(numpy.uint8)


def plot_prediction(
    img_array: numpy.ndarray,
    labels: Sequence,
    max_pred: Sequence,
    pred: Sequence,
    n_col: int = 3,
) -> None:
    """

    Args:
      img_array:
      labels:
      max_pred:
      pred:
      n_col:
    """
    n_row = len(img_array) // n_col

    f, plots = pyplot.subplots(
        n_row, n_col, sharex="all", sharey="all", figsize=(n_col * 4, n_row * 4)
    )

    for i in range(len(img_array)):
        plots[i // n_col, i % n_col].imshow(img_array[i])
        plots[i // n_col, i % n_col].set_title(
            f"truth:{labels[i]},\n max_pred:{max_pred[i]},\n pred:{pred[i]}", fontsize=8
        )


def bounding_box_from_mask(
    hard_mask: numpy.ndarray,
) -> Tuple[Number, Number, Number, Number]:
    """

    Args:
      hard_mask:

    Returns:

    """
    nz = numpy.nonzero(hard_mask)
    return numpy.min(nz[0]), numpy.min(nz[1]), numpy.max(nz[0]), numpy.max(nz[1])
