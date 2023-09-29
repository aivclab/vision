#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/03/2020
           """

import argparse
from typing import List

import cv2
import numpy
import torch
from PIL import ImageFont
from draugr.numpy_utilities import SplitEnum
from draugr.opencv_utilities import draw_bounding_boxes, frame_generator, show_image
from draugr.torch_utilities import (
    TorchDeviceSession,
    TorchEvalSession,
    global_torch_device,
)
from draugr.visualisation import progress_bar
from warg import NOD

from neodroidvision import PACKAGE_DATA_PATH, PROJECT_APP_PATH
from neodroidvision.detection import SSDOut
from neodroidvision.detection.single_stage.ssd.bounding_boxes.ssd_transforms import (
    SSDTransform,
)


@torch.no_grad()
def run_traced_webcam_demo(
    input_cfg: NOD,
    categories: List,
    score_threshold: float = 0.7,
    window_name: str = "SSD",
    onnx_exported: bool = False,
):
    """

    :param onnx_exported:
    :type onnx_exported:
    :param input_cfg:
    :type input_cfg:
    :param categories:
    :type categories:
    :param score_threshold:
    :type score_threshold:
    :param window_name:
    :type window_name:
    :return:
    :rtype:"""

    pass
    import torch

    cpu_device = torch.device("cpu")
    transforms = SSDTransform(
        input_cfg.image_size, input_cfg.pixel_mean, split=SplitEnum.testing
    )
    model = None

    if onnx_exported:
        import onnx

        onnx_model = onnx.load("torch_model.onnx")
        onnx.checker.check_model(onnx_model)

        import onnxruntime

        ort_session = onnxruntime.InferenceSession("torch_model.onnx")

        def to_numpy(tensor):
            """

            Args:
              tensor:

            Returns:

            """
            return (
                tensor.detach().cpu().numpy()
                if tensor.requires_grad
                else tensor.cpu().numpy()
            )

        x = None

        # compute onnxruntime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
        ort_outs = ort_session.run(None, ort_inputs)
    else:
        import torch
        import io

        torch.jit.load("torch_model.traced")

        with open(
            "torch_model.traced", "rb"
        ) as f:  # Load ScriptModule from io.BytesIO object
            buffer = io.BytesIO(f.read())

        model = torch.jit.load(buffer)  # Load all tensors to the original device

        """

buffer.seek(0)
torch.jit.load(buffer, map_location=torch.device('cpu'))     # Load all tensors onto CPU, using a device


buffer.seek(0)
model = torch.jit.load(buffer, map_location='cpu')     # Load all tensors onto CPU, using a string

# Load with extra files.
extra_files = torch._C.ExtraFilesMap()
extra_files['foo.txt'] = 'bar'
torch.jit.load('torch_model.traced', _extra_files=extra_files)
print(extra_files['foo.txt'])
#exit(0)
"""

    with TorchDeviceSession(device=global_torch_device("cpu"), model=model):
        with TorchEvalSession(model):
            for image in progress_bar(frame_generator(cv2.VideoCapture(0))):
                result = SSDOut(
                    *model(transforms(image)[0].unsqueeze(0).to(global_torch_device()))
                )
                height, width, *_ = image.shape

                result.boxes[:, 0::2] *= width / result.img_width.cpu().item()
                result.boxes[:, 1::2] *= height / result.img_height.cpu().item()
                (boxes, labels, scores) = (
                    result.boxes.to(cpu_device).numpy(),
                    result.labels.to(cpu_device).numpy(),
                    result.scores.to(cpu_device).numpy(),
                )

                indices = scores > score_threshold

                if show_image(
                    draw_bounding_boxes(
                        image,
                        boxes[indices],
                        labels=labels[indices],
                        scores=scores[indices],
                        categories=categories,
                        score_font=ImageFont.truetype(
                            PACKAGE_DATA_PATH / "Lato-Regular.ttf",
                            24,
                        ),
                    ).astype(numpy.uint8),
                    window_name,
                    wait=1,
                ):
                    break  # esc to quit


def main():
    """description"""
    from configs.mobilenet_v2_ssd320_voc0712 import base_cfg

    # from configs.efficient_net_b3_ssd300_voc0712 import base_cfg
    # from configs.vgg_ssd300_coco_trainval35k import base_cfg
    # from .configs.vgg_ssd512_coco_trainval35k import base_cfg

    global_torch_device(override=global_torch_device("cpu"))

    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=PROJECT_APP_PATH.user_data
        / "ssd"
        / "models"
        / "mobilenet_v2_ssd320_voc0712.pth"
        # "mobilenet_v2_ssd320_voc0712.pth"
        # "vgg_ssd300_coco_trainval35k.pth"
        # "vgg_ssd512_coco_trainval35k.pth"
        ,
        help="Use weights from path",
    )
    parser.add_argument("--score_threshold", type=float, default=0.7)
    args = parser.parse_args()

    run_traced_webcam_demo(
        input_cfg=base_cfg.input,
        categories=base_cfg.dataset_type.response_shape,
        score_threshold=args.score_threshold,
    )


if __name__ == "__main__":
    main()
