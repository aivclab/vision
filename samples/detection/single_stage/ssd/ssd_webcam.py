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
import pkg_resources
import torch
from PIL import ImageFont
from apppath import ensure_existence
from neodroidvision import PACKAGE_DATA_PATH, PROJECT_APP_PATH, PROJECT_NAME
from neodroidvision.detection import SingleShotDectectionNms
from neodroidvision.detection.single_stage.ssd.bounding_boxes.ssd_transforms import (
    SSDTransform,
)
from neodroidvision.utilities import CheckPointer
from tqdm import tqdm
from warg import NOD

from draugr.opencv_utilities import draw_bounding_boxes, frame_generator
from draugr.torch_utilities import Split, TorchEvalSession, global_torch_device


@torch.no_grad()
def run_webcam_demo(
    cfg: NOD,
    input_cfg: NOD,
    categories: List,
    model_ckpt: Path,
    score_threshold: float = 0.7,
    window_name: str = "SSD",
):
    """

:param categories:
:type categories:
:param cfg:
:type cfg:
:param model_ckpt:
:type model_ckpt:
:param score_threshold:
:type score_threshold:
:param window_name:
:type window_name:
:return:
:rtype:
"""

    cpu_device = torch.device("cpu")
    transforms = SSDTransform(
        input_cfg.image_size, input_cfg.pixel_mean, split=Split.Testing
    )
    model = SingleShotDectectionNms(cfg)

    checkpointer = CheckPointer(
        model, save_dir=ensure_existence(PROJECT_APP_PATH.user_data / "results")
    )
    checkpointer.load(model_ckpt, use_latest=model_ckpt is None)
    print(
        f"Loaded weights from {model_ckpt if model_ckpt else checkpointer.get_checkpoint_file()}"
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

            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(
                window_name,
                draw_bounding_boxes(
                    image,
                    boxes[indices],
                    labels=labels[indices],
                    scores=scores[indices],
                    categories=categories,
                    score_font=ImageFont.truetype(
                        PACKAGE_DATA_PATH/"Lato-Regular.ttf",
                        24,
                    ),
                ).astype(numpy.uint8),
            )
            if cv2.waitKey(1) == 27:
                break  # esc to quit


def main():
    from configs.vgg_ssd300_coco_trainval35k import base_cfg

    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=PROJECT_APP_PATH.user_data / "ssd" / "models" /
                "mobilenet_v2_ssd320_voc0712.pth"
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
        categories=base_cfg.dataset_type.category_sizes,
        model_ckpt=Path(args.ckpt),
        score_threshold=args.score_threshold,
    )


if __name__ == "__main__":
    main()
