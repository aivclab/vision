from itertools import cycle
from typing import Iterable, Optional, Sequence, Tuple, Union

import PIL
import numpy
import six
from PIL import Image, ImageColor, ImageDraw, ImageFont
from attr import dataclass
from draugr.pandas_utilities import ColorEnum
from draugr.visualisation.pillow_utilities import PilModesEnum
from matplotlib import pyplot
from warg import Number
from warg.mixins.dict_mixins import IterDictValuesMixin

__author__ = "Christian Heider Nielsen"
__doc__ = r"""A set of functions that are used for visualisation.

These functions often receive an image, perform some visualisation on the image.
The functions do not return a value, instead they modify the image itself.

"""


@dataclass
class BoundingBoxCoordinatesSpec(IterDictValuesMixin):
    x_min = 0
    y_min = 0
    x_max = 0
    y_max = 0

    """
def __init__(self,x_min,y_min,x_max,y_max):
self.x_min = x_min
self.y_min = y_min
self.x_max = x_max
self.y_max = y_max
"""


@dataclass
class BoundingBoxSpec:
    __slots__ = ["coordinates", "score", "label", "mask", "keypoints", "color"]

    coordinates: Tuple[float]
    score: float
    label: str
    mask: Tuple[float]
    keypoints: Tuple[float]
    color: Union[str, tuple]

    def __post_init__(self):
        pass


_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10
STANDARD_COLORS = [
    "AliceBlue",
    "Chartreuse",
    "Aqua",
    "Aquamarine",
    "Azure",
    "Beige",
    "Bisque",
    "BlanchedAlmond",
    "BlueViolet",
    "BurlyWood",
    "CadetBlue",
    "AntiqueWhite",
    "Chocolate",
    "Coral",
    "CornflowerBlue",
    "Cornsilk",
    "Crimson",
    "Cyan",
    "DarkCyan",
    "DarkGoldenRod",
    "DarkGrey",
    "DarkKhaki",
    "DarkOrange",
    "DarkOrchid",
    "DarkSalmon",
    "DarkSeaGreen",
    "DarkTurquoise",
    "DarkViolet",
    "DeepPink",
    "DeepSkyBlue",
    "DodgerBlue",
    "FireBrick",
    "FloralWhite",
    "ForestGreen",
    "Fuchsia",
    "Gainsboro",
    "GhostWhite",
    "Gold",
    "GoldenRod",
    "Salmon",
    "Tan",
    "HoneyDew",
    "HotPink",
    "IndianRed",
    "Ivory",
    "Khaki",
    "Lavender",
    "LavenderBlush",
    "LawnGreen",
    "LemonChiffon",
    "LightBlue",
    "LightCoral",
    "LightCyan",
    "LightGoldenRodYellow",
    "LightGray",
    "LightGrey",
    "LightGreen",
    "LightPink",
    "LightSalmon",
    "LightSeaGreen",
    "LightSkyBlue",
    "LightSlateGray",
    "LightSlateGrey",
    "LightSteelBlue",
    "LightYellow",
    "Lime",
    "LimeGreen",
    "Linen",
    "Magenta",
    "MediumAquaMarine",
    "MediumOrchid",
    "MediumPurple",
    "MediumSeaGreen",
    "MediumSlateBlue",
    "MediumSpringGreen",
    "MediumTurquoise",
    "MediumVioletRed",
    "MintCream",
    "MistyRose",
    "Moccasin",
    "NavajoWhite",
    "OldLace",
    "Olive",
    "OliveDrab",
    "Orange",
    "OrangeRed",
    "Orchid",
    "PaleGoldenRod",
    "PaleGreen",
    "PaleTurquoise",
    "PaleVioletRed",
    "PapayaWhip",
    "PeachPuff",
    "Peru",
    "Pink",
    "Plum",
    "PowderBlue",
    "Purple",
    "Red",
    "RosyBrown",
    "RoyalBlue",
    "SaddleBrown",
    "Green",
    "SandyBrown",
    "SeaGreen",
    "SeaShell",
    "Sienna",
    "Silver",
    "SkyBlue",
    "SlateBlue",
    "SlateGray",
    "SlateGrey",
    "Snow",
    "SpringGreen",
    "SteelBlue",
    "GreenYellow",
    "Teal",
    "Thistle",
    "Tomato",
    "Turquoise",
    "Violet",
    "Wheat",
    "White",
    "WhiteSmoke",
    "Yellow",
    "YellowGreen",
]


def encode_image_array_as_png_str(image: PIL.Image) -> bytes:
    """Encodes a numpy array into a PNG string.

    Args:
    image: a numpy array with shape [height, width, 3].

    Returns:
    PNG encoded image string.
    """
    image_pil = Image.fromarray(numpy.uint8(image))
    output = six.BytesIO()
    image_pil.save(output, format="PNG")
    png_string = output.getvalue()
    output.close()
    return png_string


def draw_bounding_box_on_image_array(
    image: numpy.ndarray,
    y_min: int,
    x_min: int,
    y_max: int,
    x_max: int,
    labels: tuple = (),
    *,
    color: ColorEnum = None,
    thickness: int = 2,
    use_normalized_coordinates: bool = True,
    mode: PilModesEnum = PilModesEnum.RGBA,
) -> None:
    """Adds a bounding box to an image (numpy array).

    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    Args:
    image: a numpy array with shape [height, width, 3].
    y_min: y_min of bounding box.
    x_min: x_min of bounding box.
    y_max: y_max of bounding box.
    x_max: x_max of bounding box.
    color: color to draw bounding box.
    thickness: line thickness. Default value is 2.
    labels: list of strings to display in box
                (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
    y_min, x_min, y_max, x_max as relative to the image.  Otherwise treat
    coordinates as absolute."""

    image_pil = Image.fromarray(image, mode=mode)
    draw_bounding_box_on_image(
        image_pil,
        y_min,
        x_min,
        y_max,
        x_max,
        labels,
        line_color=color,
        thickness=thickness,
        use_normalized_coordinates=use_normalized_coordinates,
    )
    numpy.copyto(image, numpy.array(image_pil))


def draw_bounding_box_on_image(
    image: PIL.Image.Image,
    x_min: Number,
    y_min: Number,
    x_max: Number,
    y_max: Number,
    labels: Optional[Iterable] = (),
    *,
    line_color: ColorEnum = None,
    thickness: Number = 2,
    use_normalized_coordinates=True,
    label_inside: bool = True,
    text_color: ColorEnum = None,
):
    """Adds a bounding box to an image.

    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input 'color'.
    If the top of the bounding box extends to the edge of the image, the strings
    are displayed below the bounding box.

    Args:
    :param image: a PIL.Image object.

    :param x_min: x_min of bounding box.
    :param y_min: y_min of bounding box.
    :param x_max: x_max of bounding box.
    :param y_max: y_max of bounding box.

    :param line_color: color to draw bounding box.
    :param thickness: line thickness. Default value is 2.
    :param labels: list of strings to display in box
                (each to be shown on its own line).
    :param use_normalized_coordinates: If True (default), treat coordinates
    y_min, x_min, y_max, x_max as relative to the image.  Otherwise treat
    coordinates as absolute.

    :param label_inside:
    :param text_color:"""
    draw = ImageDraw.Draw(image)
    if line_color:
        line_color = ColorEnum(line_color)
    else:
        line_color = ColorEnum.red

    if text_color:
        text_color = ColorEnum(text_color)
    else:
        if line_color == ColorEnum.white:
            text_color = ColorEnum.black
        else:
            text_color = ColorEnum.white

    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (
            x_min * im_width,
            x_max * im_width,
            y_min * im_height,
            y_max * im_height,
        )
    else:
        (left, right, top, bottom) = (x_min, x_max, y_min, y_max)
    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
        width=thickness,
        fill=line_color.value,
    )
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    if labels:
        # If the total height of the display strings added to the top of the bounding
        # box exceeds the top of the image, stack the strings below the bounding box
        # instead of above.
        display_str_size = [font.getsize(ds) for ds in labels]
        display_str_width, display_str_height = zip(*display_str_size)
        # Each display_str has a top and bottom margin of 0.05x.
        total_display_str_height = (1 + 2 * 0.05) * sum(display_str_height)
        total_display_str_width = sum(display_str_width)

        if left < 0:
            text_left = right - total_display_str_width
        else:
            text_left = left

        if top > total_display_str_height:

            if label_inside:
                text_bottom = top + total_display_str_height
            else:
                text_bottom = top
        else:
            if label_inside:
                text_bottom = bottom
            else:
                text_bottom = bottom + total_display_str_height
        # Reverse list and print from bottom to top.
        for display_str in labels[::-1]:
            text_width, text_height = font.getsize(display_str)
            margin = numpy.ceil(0.05 * text_height)
            draw.rectangle(
                [
                    (text_left, text_bottom - text_height - 2 * margin),
                    (text_left + text_width, text_bottom),
                ],
                fill=line_color.value,
            )
            draw.text(
                (text_left + margin, text_bottom - text_height - margin),
                display_str,
                fill=text_color.value,
                font=font,
            )
            text_bottom -= text_height - 2 * margin


def draw_bounding_boxes_on_image_array(
    image,
    boxes,
    labels: Optional[Iterable] = None,
    *,
    color: Optional[ColorEnum] = None,
    thickness: Number = 2,
    mode: PilModesEnum = PilModesEnum.RGBA,
) -> None:
    """Draws bounding boxes on image (numpy array).

    Args:
    :param image: a numpy array object.
    :param boxes: a 2 dimensional numpy array of [N, 4]: (y_min, x_min, y_max, x_max).
     The coordinates are in normalized format between [0, 1].
    :param color: color to draw bounding box.
    :param thickness: line thickness. Default value is 4.
    :param labels: list of list of strings.
                     a list of strings for each bounding box.
                     The reason to pass a list of strings for a
                     bounding box is that it might contain
                     multiple labels.
    :param mode:
    Raises:
    ValueError: if boxes is not a [N, 4] array"""
    image_pil = Image.fromarray(image, mode=mode.value)
    draw_bounding_boxes_on_image(
        image_pil, boxes, labels, color=color, thickness=thickness
    )
    numpy.copyto(image, numpy.array(image_pil) / 255.0)


def draw_bounding_boxes_on_image(
    image,
    boxes,
    labels_iterable: Optional[Iterable] = None,
    *,
    color: ColorEnum = None,
    thickness: Number = 2,
):
    """Draws bounding boxes on image.

    Args:
    image: a PIL.Image object.
    boxes: a 2 dimensional numpy array of [N, 4]: (y_min, x_min, y_max, x_max).
     The coordinates are in normalized format between [0, 1].
    color: color to draw bounding box.
    thickness: line thickness. Default value is 4.
    labels_iterable: list of list of strings.
                     a list of strings for each bounding box.
                     The reason to pass a list of strings for a
                     bounding box is that it might contain
                     multiple labels.

    Raises:
    ValueError: if boxes is not a [N, 4] array"""
    boxes_shape = boxes.shape
    color_iter = None
    if not color:
        if labels_iterable:
            labels_iterable = iter(labels_iterable)
            color_iter = iter(cycle([*ColorEnum]))
        else:
            color = ColorEnum.red
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError("Input must be of size [N, 4]")
    for i in range(boxes_shape[0]):
        labels = ()
        if labels_iterable:
            labels = next(labels_iterable)
            if color_iter:
                color = next(color_iter)

        draw_bounding_box_on_image(
            image,
            boxes[i, 0],
            boxes[i, 1],
            boxes[i, 2],
            boxes[i, 3],
            labels,
            line_color=color,
            thickness=thickness,
        )


def _visualize_boxes(image, boxes, classes, scores, category_index, **kwargs):
    return visualize_boxes_and_labels_on_image_array(
        image, boxes, classes, scores, category_index=category_index, **kwargs
    )


def _visualize_boxes_and_masks(
    image, boxes, classes, scores, masks, category_index, **kwargs
):
    return visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes,
        scores,
        category_index=category_index,
        instance_masks=masks,
        **kwargs,
    )


def _visualize_boxes_and_keypoints(
    image, boxes, classes, scores, keypoints, category_index, **kwargs
):
    return visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes,
        scores,
        category_index=category_index,
        keypoints=keypoints,
        **kwargs,
    )


def _visualize_boxes_and_masks_and_keypoints(
    image, boxes, classes, scores, masks, keypoints, category_index, **kwargs
):
    return visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes,
        scores,
        category_index=category_index,
        instance_masks=masks,
        keypoints=keypoints,
        **kwargs,
    )


def draw_keypoints_on_image_array(
    image,
    keypoints,
    color: ColorEnum = None,
    radius: Number = 2,
    use_normalized_coordinates: bool = True,
    mode: PilModesEnum = PilModesEnum.RGBA,
):
    """Draws keypoints on an image (numpy array).

    Args:
    image: a numpy array with shape [height, width, 3].
    keypoints: a numpy array with shape [num_keypoints, 2].
    color: color to draw the keypoints with.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
    relative to the image.  Otherwise treat them as absolute.
    :param use_normalized_coordinates:
    :param radius:
    :param color:
    :param keypoints:
    :param image:
    :param mode:"""
    image_pil = Image.fromarray(image, mode=mode)
    draw_keypoints_on_image(
        image_pil, keypoints, color, radius, use_normalized_coordinates
    )
    numpy.copyto(image, numpy.array(image_pil))


def draw_keypoints_on_image(
    image,
    keypoints,
    color: ColorEnum = None,
    radius: Number = 2,
    use_normalized_coordinates: bool = True,
):
    """Draws keypoints on an image.

    Args:
    image: a PIL.Image object.
    keypoints: a numpy array with shape [num_keypoints, 2].
    color: color to draw the keypoints with.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
    relative to the image.  Otherwise treat them as absolute."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    keypoints_x = [k[1] for k in keypoints]
    keypoints_y = [k[0] for k in keypoints]
    if use_normalized_coordinates:
        keypoints_x = tuple([im_width * x for x in keypoints_x])
        keypoints_y = tuple([im_height * y for y in keypoints_y])
    for keypoint_x, keypoint_y in zip(keypoints_x, keypoints_y):
        draw.ellipse(
            [
                (keypoint_x - radius, keypoint_y - radius),
                (keypoint_x + radius, keypoint_y + radius),
            ],
            outline=color,
            fill=color,
        )


def draw_mask_on_image_array(
    image,
    mask,
    color: ColorEnum = None,
    alpha: float = 0.4,
    mode: PilModesEnum = PilModesEnum.RGBA,
):
    """Draws mask on an image.

    Args:
    image: uint8 numpy array with shape (img_height, img_height, 3)
    mask: a uint8 numpy array of shape (img_height, img_height) with
    values between either 0 or 1.
    color: color to draw the keypoints with.
    alpha: transparency value between 0 and 1. (default: 0.4)

    Raises:
    ValueError: On incorrect data type for image or masks.
    :param alpha:
    :param color:
    :param mask:
    :param image:
    :param mode:"""
    if image.dtype != numpy.uint8:
        raise ValueError("`image` not of type numpy.uint8")
    if mask.dtype != numpy.uint8:
        raise ValueError("`mask` not of type numpy.uint8")
    if numpy.any(numpy.logical_and(mask != 1, mask != 0)):
        raise ValueError("`mask` elements should be in [0, 1]")
    if image.shape[:2] != mask.shape:
        raise ValueError(
            "The image has spatial dimensions %s but the mask has "
            "dimensions %s" % (image.shape[:2], mask.shape)
        )
    rgb = ImageColor.getrgb(color)
    pil_image = Image.fromarray(image, mode=mode)

    solid_color = numpy.expand_dims(numpy.ones_like(mask), axis=2) * numpy.reshape(
        list(rgb), [1, 1, 3]
    )
    pil_solid_color = Image.fromarray(numpy.uint8(solid_color)).convert("RGBA")
    pil_mask = Image.fromarray(numpy.uint8(255.0 * alpha * mask)).convert("L")
    pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
    numpy.copyto(image, numpy.array(pil_image.convert(mode)))


def visualize_boxes_and_labels_on_image_array(
    image,
    bounding_boxes: Sequence[BoundingBoxSpec],
    use_normalized_coordinates=True,
    max_boxes_to_draw=20,
    min_score_thresh=0.5,
    line_thickness=2,
):
    """Overlay labeled boxes on an image with formatted scores and label names.

    This function groups boxes that correspond to the same location
    and creates a display string for each detection and overlays these
    on the image. Note that this function modifies the image in place, and returns
    that same image.

    Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
    and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
    this function assumes that the boxes to be plotted are groundtruth
    boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
    category index `id` and category name `name`) keyed by category indices.
    instance_masks: a numpy array of shape [N, image_height, image_width] with
    values ranging between 0 and 1, can be None.
    instance_boundaries: a numpy array of shape [N, image_height, image_width]
    with values ranging between 0 and 1, can be None.
    keypoints: a numpy array of shape [N, num_keypoints, 2], can
    be None
    use_normalized_coordinates: whether boxes is to be interpreted as
    normalized coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
    all boxes.
    min_score_thresh: minimum score threshold for a box to be visualized
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
    class-agnostic mode or not.  This mode will display scores but ignore
    classes.
    line_thickness: integer (default: 2) controlling line width of the boxes.
    groundtruth_box_visualization_color: box color for visualizing groundtruth
    boxes
    skip_scores: whether to skip score when drawing a single detection
    skip_labels: whether to skip label when drawing a single detection

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
    :param bounding_boxes:"""
    # Create a display string (and color) for every box location, group any boxes
    # that correspond to the same location.

    if not max_boxes_to_draw:
        max_boxes_to_draw = len(bounding_boxes)

    for i in range(min(max_boxes_to_draw, len(bounding_boxes))):
        box = bounding_boxes[i]
        if box.label is None:
            box.label = "None"
        if box.color is None:
            box.color = STANDARD_COLORS[i % len(STANDARD_COLORS)]

        if box.score is None or box.score > min_score_thresh:
            if box.mask is not None:
                draw_mask_on_image_array(image, box.mask, color=box.color, alpha=1.0)

            draw_bounding_box_on_image_array(
                image,
                *box.coordinates,
                color=box.color,
                thickness=line_thickness,
                labels=box.label,
                use_normalized_coordinates=use_normalized_coordinates,
            )

            if box.keypoints is not None:
                draw_keypoints_on_image_array(
                    image,
                    box.keypoints,
                    color=box.color,
                    radius=line_thickness // 2,
                    use_normalized_coordinates=use_normalized_coordinates,
                )

    return image


def hist_plot(values, bins):
    """Numpy function to plot hist."""
    fig = pyplot.figure(frameon=False)
    ax = fig.add_subplot("111")
    y, x = numpy.histogram(values, bins=bins)
    ax.plot(x[:-1], y)
    ax.set_ylabel("count")
    ax.set_xlabel("value")
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = numpy.fromstring(fig.canvas.tostring_rgb(), dtype="uint8").reshape(
        (1, int(height), int(width), 3)
    )
    return image


def cdf_plot(values):
    """Numpy function to plot CDF."""
    normalized_values = values / numpy.sum(values)
    sorted_values = numpy.sort(normalized_values)
    cumulative_values = numpy.cumsum(sorted_values)
    fraction_of_examples = (
        numpy.arange(cumulative_values.size, dtype=numpy.float32)
        / cumulative_values.size
    )
    fig = pyplot.figure(frameon=False)
    ax = fig.add_subplot("111")
    ax.plot(fraction_of_examples, cumulative_values)
    ax.set_ylabel("cumulative normalized values")
    ax.set_xlabel("fraction of examples")
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = numpy.fromstring(fig.canvas.tostring_rgb(), dtype="uint8").reshape(
        (1, int(height), int(width), 3)
    )
    return image


if __name__ == "__main__":

    def main():
        """ """
        im = numpy.random.rand(256, 256, 4)
        bb = numpy.array(
            [
                [0.1, 0.1, 0.9, 0.9],
                [0.2, 0.2, 0.8, 0.8],
                [0.3, 0.3, 0.7, 0.7],
                [0.4, 0.4, 0.6, 0.6],
                [0.5, 0.5, 0.5, 0.5],
            ]
        )

        labels_str = (*numpy.arange(0, 5).astype(str).tolist(),)

        draw_bounding_boxes_on_image_array(im, bb, labels_str)
        # bs = [BoundingBoxSpec(bb_,l_,None,None,None,'white') for bb_, l_ in zip(bb,labels_str)]

        # visualize_boxes_and_labels_on_image_array(im, bs)
        pyplot.imshow(im)
        pyplot.show()

    def main2():
        """ """
        im = numpy.random.rand(256, 256, 3)
        bb = numpy.array(
            [
                [0.1, 0.1, 0.9, 0.9],
                [0.2, 0.2, 0.8, 0.8],
                [0.3, 0.3, 0.7, 0.7],
                [0.4, 0.4, 0.6, 0.6],
                [0.5, 0.5, 0.5, 0.5],
            ]
        )

        labels_str = (*numpy.arange(0, 5).astype(str).tolist(),)

        draw_bounding_boxes_on_image_array(im, bb, labels_str, mode=PilModesEnum.RGB)
        # bs = [BoundingBoxSpec(bb_,l_,None,None,None,'white') for bb_, l_ in zip(bb,labels_str)]

        # visualize_boxes_and_labels_on_image_array(im, bs)
        pyplot.imshow(im)
        pyplot.show()

    def main3():
        """ """
        im = numpy.random.rand(256, 256)
        bb = numpy.array(
            [
                [0.1, 0.1, 0.9, 0.9],
                [0.2, 0.2, 0.8, 0.8],
                [0.3, 0.3, 0.7, 0.7],
                [0.4, 0.4, 0.6, 0.6],
                [0.5, 0.5, 0.5, 0.5],
            ]
        )

        labels_str = (*numpy.arange(0, 5).astype(str).tolist(),)

        draw_bounding_boxes_on_image_array(im, bb, labels_str, mode=PilModesEnum.L)
        # bs = [BoundingBoxSpec(bb_,l_,None,None,None,'white') for bb_, l_ in zip(bb,labels_str)]

        # visualize_boxes_and_labels_on_image_array(im, bs)
        pyplot.imshow(im)
        pyplot.show()

    main2()
