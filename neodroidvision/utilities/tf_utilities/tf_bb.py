import functools
from pathlib import Path

import numpy
import tensorflow
from PIL.Image import Image, fromarray
from utilities.visualisation.bounding_box_visualisation import (
    _visualize_boxes,
    _visualize_boxes_and_keypoints,
    _visualize_boxes_and_masks,
    _visualize_boxes_and_masks_and_keypoints,
    cdf_plot,
    hist_plot,
)


def save_image_array_as_png(image: Image, output_path: Path) -> None:
    """Saves an image (represented as a numpy array) to PNG.

    Args:
    image: a numpy array with shape [height, width, 3].
    output_path: path to which image should be written.
    """
    image_pil = fromarray(numpy.uint8(image)).convert("RGB")
    with tensorflow.gfile.Open(output_path, "w") as fid:
        image_pil.save(fid, "PNG")


def draw_bounding_boxes_on_image_tensors(
    images,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    keypoints=None,
    max_boxes_to_draw=20,
    min_score_thresh=0.2,
    line_thickness=2,
):
    """Draws bounding boxes, masks, and keypoints on batch of image tensors.

    Args:
    :param images: A 4D uint8 image tensor of shape [N, H, W, C].
    :param boxes: [N, max_detections, 4] float32 tensor of detection boxes.
    :param classes: [N, max_detections] int tensor of detection classes. Note that
    classes are 1-indexed.
    :param scores: [N, max_detections] float32 tensor of detection scores.
    :param category_index: a dict that maps integer ids to category dicts. e.g.
    {1: {1: 'dog'}, 2: {2: 'cat'}, ...}
    :param instance_masks: A 4D uint8 tensor of shape [N, max_detection, H, W] with
    instance masks.
    :param keypoints: A 4D float32 tensor of shape [N, max_detection, num_keypoints, 2]
    with keypoints.
    :param max_boxes_to_draw: Maximum number of boxes to draw on an image. Default 20.
    :param min_score_thresh: Minimum score threshold for visualization. Default 0.2.
    :param line_thickness:

    Returns:
    4D image tensor of type uint8, with boxes drawn on top.

    """
    visualization_keyword_args = {
        "use_normalized_coordinates": True,
        "max_boxes_to_draw": max_boxes_to_draw,
        "min_score_thresh": min_score_thresh,
        "agnostic_mode": False,
        "line_thickness": line_thickness,
    }

    if instance_masks is not None and keypoints is None:
        visualize_boxes_fn = functools.partial(
            _visualize_boxes_and_masks,
            category_index=category_index,
            **visualization_keyword_args,
        )
        elems = [images, boxes, classes, scores, instance_masks]
    elif instance_masks is None and keypoints is not None:
        visualize_boxes_fn = functools.partial(
            _visualize_boxes_and_keypoints,
            category_index=category_index,
            **visualization_keyword_args,
        )
        elems = [images, boxes, classes, scores, keypoints]
    elif instance_masks is not None and keypoints is not None:
        visualize_boxes_fn = functools.partial(
            _visualize_boxes_and_masks_and_keypoints,
            category_index=category_index,
            **visualization_keyword_args,
        )
        elems = [images, boxes, classes, scores, instance_masks, keypoints]
    else:
        visualize_boxes_fn = functools.partial(
            _visualize_boxes,
            category_index=category_index,
            **visualization_keyword_args,
        )
        elems = [images, boxes, classes, scores]

    def draw_boxes(image_and_detections):
        """Draws boxes on image."""
        image_with_boxes = tensorflow.py_func(
            visualize_boxes_fn, image_and_detections, tensorflow.uint8
        )
        return image_with_boxes

    images = tensorflow.map_fn(
        draw_boxes, elems, dtype=tensorflow.uint8, back_prop=False
    )
    return images


def add_cdf_image_summary(values, name):
    """Adds a tf.summary.image for a CDF plot of the values.

    Normalizes `values` such that they sum to 1, plots the cumulative distribution
    function and creates a tf image summary.

    Args:
    values: a 1-D float32 tensor containing the values.
    name: name for the image summary."""

    tensorflow.summary.image(
        name, tensorflow.py_func(cdf_plot, [values], tensorflow.uint8)
    )


def add_hist_image_summary(values, bins, name):
    """Adds a tf.summary.image for a histogram plot of the values.

    Plots the histogram of values and creates a tf image summary.

    Args:
    values: a 1-D float32 tensor containing the values.
    bins: bin edges which will be directly passed to numpy.histogram.
    name: name for the image summary."""

    tensorflow.summary.image(
        name, tensorflow.py_func(hist_plot, [values, bins], tensorflow.uint8)
    )
