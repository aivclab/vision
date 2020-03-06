#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 04/03/2020
           """

import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import cv2
import numpy
from PIL import Image

try:
    FONT = ImageFont.truetype("arial.ttf", 24)
except IOError:
    FONT = ImageFont.load_default()

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def find_contours(*args, **kwargs):
    """
  Wraps cv2.findContours to maintain compatibility between versions 3 and 4
  Returns:
      contours, hierarchy
  """
    if cv2.__version__.startswith("4"):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif cv2.__version__.startswith("3"):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError("cv2 must be either version 3 or 4 to call this method")
    return contours, hierarchy


def compute_color_for_labels(label):
    """
  Simple function that adds fixed color depending on the class
  """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def _draw_single_box(
    image,
    xmin,
    ymin,
    xmax,
    ymax,
    color=(0, 255, 0),
    display_str=None,
    font=None,
    width=2,
    alpha=0.5,
    fill=False,
):
    if font is None:
        font = FONT

    draw = ImageDraw.Draw(image, mode="RGBA")
    left, right, top, bottom = xmin, xmax, ymin, ymax
    alpha_color = color + (int(255 * alpha),)
    draw.rectangle(
        [(left, top), (right, bottom)],
        outline=color,
        fill=alpha_color if fill else None,
        width=width,
    )

    if display_str:
        text_bottom = bottom
        # Reverse list and print from bottom to top.
        text_width, text_height = font.getsize(display_str)
        margin = numpy.ceil(0.05 * text_height)
        draw.rectangle(
            xy=[
                (left + width, text_bottom - text_height - 2 * margin - width),
                (left + text_width + width, text_bottom - width),
            ],
            fill=alpha_color,
        )
        draw.text(
            (left + margin + width, text_bottom - text_height - margin - width),
            display_str,
            fill="black",
            font=font,
        )

    return image


def draw_boxes(
    image,
    boxes,
    labels=None,
    scores=None,
    class_name_map=None,
    width=2,
    alpha=0.5,
    fill=False,
    font=None,
    score_format=":{:.2f}",
) -> numpy.ndarray:
    """Draw bboxes(labels, scores) on image
  Args:
      image: numpy array image, shape should be (height, width, channel)
      boxes: bboxes, shape should be (N, 4), and each row is (xmin, ymin, xmax, ymax)
      labels: labels, shape: (N, )
      scores: label scores, shape: (N, )
      class_name_map: list or dict, map class id to class name for visualization.
      width: box width
      alpha: text background alpha
      fill: fill box or not
      font: text font
      score_format: score format
  Returns:
      An image with information drawn on it.
  """
    boxes = numpy.array(boxes)
    num_boxes = boxes.shape[0]
    if isinstance(image, Image.Image):
        draw_image = image
    elif isinstance(image, numpy.ndarray):
        draw_image = Image.fromarray(image)
    else:
        raise AttributeError("Unsupported images type {}".format(type(image)))

    for i in range(num_boxes):
        display_str = ""
        color = (0, 255, 0)
        if labels is not None:
            this_class = labels[i]
            color = compute_color_for_labels(this_class)
            class_name = (
                class_name_map[this_class]
                if class_name_map is not None
                else str(this_class)
            )
            display_str = class_name

        if scores is not None:
            prob = scores[i]
            if display_str:
                display_str += score_format.format(prob)
            else:
                display_str += f"score{score_format.format(prob)}"

        draw_image = _draw_single_box(
            image=draw_image,
            xmin=boxes[i, 0],
            ymin=boxes[i, 1],
            xmax=boxes[i, 2],
            ymax=boxes[i, 3],
            color=color,
            display_str=display_str,
            font=font,
            width=width,
            alpha=alpha,
            fill=fill,
        )

    image = numpy.array(draw_image, dtype=numpy.uint8)
    return image


def draw_masks(
    image,
    masks,
    labels=None,
    border=True,
    border_width=2,
    border_color=(255, 255, 255),
    alpha=0.5,
    color=None,
) -> numpy.ndarray:
    """
  Args:
      image: numpy array image, shape should be (height, width, channel)
      masks: (N, 1, Height, Width)
      labels: mask label
      border: draw border on mask
      border_width: border width
      border_color: border color
      alpha: mask alpha
      color: mask color
  Returns:
      numpy.ndarray
  """
    if isinstance(image, Image.Image):
        image = numpy.array(image)
    assert isinstance(image, numpy.ndarray)
    masks = numpy.array(masks)
    for i, mask in enumerate(masks):
        mask = mask.squeeze()[:, :, None].astype(numpy.bool)

        label = labels[i] if labels is not None else 1
        _color = compute_color_for_labels(label) if color is None else tuple(color)

        image = numpy.where(
            mask, mask * numpy.array(_color) * alpha + image * (1 - alpha), image
        )
        if border:
            contours, hierarchy = find_contours(
                mask.astype(numpy.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            image = cv2.drawContours(
                image,
                contours,
                -1,
                border_color,
                thickness=border_width,
                lineType=cv2.LINE_AA,
            )

    image = image.astype(numpy.uint8)
    return image


if __name__ == "__main__":
    from matplotlib import pyplot
    import pickle

    coco_class_name = [
        "__bg",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    name = "000000308476"
    with open(f"data/{name}.pickle", "rb") as f:
        data = pickle.load(f)
    img = Image.open(f"data/{name}.jpg")
    img = draw_masks(img, data["masks"], data["labels"])
    img = draw_boxes(
        img,
        boxes=data["boxes"],
        labels=data["labels"],
        scores=data["scores"],
        class_name_map=coco_class_name,
        score_format=":{:.4f}",
    )
    pyplot.imshow(img)
    pyplot.show()
