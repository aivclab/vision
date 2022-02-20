import sys
import warnings

import torch
import torchvision

__all__ = ["non_maximum_suppression", "batched_non_maximum_suppression"]

__doc__ = """This file is merily a wrapper to provide a custom implementation of NMS"""

if int(torchvision.__version__.split(".")[1]) >= int("0.3.0".split(".")[1]):
    nms_support = torchvision.ops.nms
else:

    print(f"torchvision version: {torchvision.__version__}" "\n nms not supported")
    try:

        import ssd_torch_extension

        nms_support = ssd_torch_extension.nms  # non_maximum_suppression
    except ImportError or ModuleNotFoundError:
        warnings.warn(
            "No NMS is available. Please upgrade torchvision to 0.3.0+ or compile c++ NMS "
            "using `cd ext & python build.py build_ext develop`"
        )
        sys.exit(-1)


def non_maximum_suppression(boxes, scores, iou_threshold: float) -> torch.Tensor:
    """Performs non-maximum suppression, run on GPU or CPU according to
    boxes's device.
    Args:
    boxes(Tensor[N, 4]): boxes in (x1, y1, x2, y2) format, use absolute coordinates(or relative coordinates)
    scores(Tensor[N]): scores
    nms_thresh(float): thresh
    Returns:
    indices kept."""
    return nms_support(boxes, scores, iou_threshold)


def my_batched_nms(boxes, scores, idxs, iou_threshold) -> torch.Tensor:
    """

    :param boxes:
    :type boxes:
    :param scores:
    :type scores:
    :param idxs:
    :type idxs:
    :param iou_threshold:
    :type iou_threshold:
    :return:
    :rtype:"""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        # strategy: in order to perform NMS independently per class.
        # we add an offset to all the boxes. The offset is dependent
        # only on the class idx, and is large enough so that boxes
        # from different classes do not overlap
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = non_maximum_suppression(boxes_for_nms, scores, iou_threshold)
    return keep


batched_support = my_batched_nms


def batched_non_maximum_suppression(boxes, scores, idxs, iou_threshold) -> torch.Tensor:
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Parameters
    ----------
    boxes : Tensor[N, 4]
    boxes where NMS will be performed. They
    are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
    scores for each one of the boxes
    idxs : Tensor[N]
    indices of the categories for each one of the boxes.
    iou_threshold : float
    discards all overlapping boxes
    with IoU < iou_threshold

    Returns
    -------
    keep : Tensor
    int64 tensor with the indices of
    the elements that have been kept by NMS, sorted
    in decreasing order of scores"""
    return batched_support(boxes, scores, idxs, iou_threshold)
