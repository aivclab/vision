#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/03/2020
           """

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy
import torch
from PIL import ImageFont
from apppath import ensure_existence
from draugr.numpy_utilities import SplitEnum
from draugr.opencv_utilities import draw_bounding_boxes, frame_generator, show_image
from draugr.torch_utilities import TorchEvalSession, global_torch_device
from tqdm import tqdm
from warg import NOD

from neodroidvision import PACKAGE_DATA_PATH, PROJECT_APP_PATH
from neodroidvision.detection import SingleShotDetectionNms
from neodroidvision.detection.single_stage.ssd.bounding_boxes.ssd_transforms import (
    SSDTransform,
)
from neodroidvision.utilities import CheckPointer


@torch.no_grad()
def run_webcam_demo(
    cfg: NOD,
    input_cfg: NOD,
    categories: List,
    model_checkpoint: Path,
    score_threshold: float = 0.7,
    window_name: str = "SSD",
):
    """

    :param categories:
    :type categories:
    :param cfg:
    :type cfg:
    :param model_checkpoint:
    :type model_checkpoint:
    :param score_threshold:
    :type score_threshold:
    :param window_name:
    :type window_name:
    :return:
    :rtype:"""

    cpu_device = torch.device("cpu")
    transforms = SSDTransform(
        input_cfg.image_size, input_cfg.pixel_mean, split=SplitEnum.testing
    )
    model = SingleShotDetectionNms(cfg)

    checkpointer = CheckPointer(
        model, save_dir=ensure_existence(PROJECT_APP_PATH.user_data / "results")
    )
    checkpointer.load(model_checkpoint, use_latest=model_checkpoint is None)
    print(
        f"Loaded weights from {model_checkpoint if model_checkpoint else checkpointer.get_checkpoint_file()}"
    )

    model.post_init()
    model.to(global_torch_device())

    with TorchEvalSession(model):
        for image in tqdm(frame_generator(cv2.VideoCapture(0))):
            result = model(transforms(image)[0].unsqueeze(0).to(global_torch_device()))
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
    """ """
    from configs.vgg_ssd300_coco_trainval35k import base_cfg

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

    run_webcam_demo(
        cfg=base_cfg,
        input_cfg=base_cfg.input,
        categories=base_cfg.dataset_type.categories,
        model_checkpoint=Path(args.ckpt),
        score_threshold=args.score_threshold,
    )


if __name__ == "__main__":
    main()
