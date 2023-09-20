#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/03/2020
           """

import argparse
import numpy
import os
import time
import torch
from PIL import Image, ImageFont
from draugr.numpy_utilities import SplitEnum
from draugr.opencv_utilities import draw_bounding_boxes
from draugr.torch_utilities import global_torch_device
from neodroidvision import PACKAGE_DATA_PATH, PROJECT_APP_PATH
from neodroidvision.detection import SingleShotDetection
from neodroidvision.detection.single_stage.ssd.bounding_boxes.ssd_transforms import (
    SSDTransform,
)
from neodroidvision.utilities import CheckPointer
from pathlib import Path
from typing import Sequence
from warg import NOD, ensure_existence


@torch.no_grad()
def run_demo(
    cfg: NOD,
    categories: Sequence,
    model_checkpoint,
    score_threshold: float,
    images_dir: Path,
    output_dir: Path,
):
    """

    Args:
      cfg:
      categories:
      model_checkpoint:
      score_threshold:
      images_dir:
      output_dir:
    """
    model = SingleShotDetection(cfg)

    checkpointer = CheckPointer(
        model, save_dir=ensure_existence(PROJECT_APP_PATH.user_data / "results")
    )
    checkpointer.load(model_checkpoint, use_latest=model_checkpoint is None)
    print(
        f"Loaded weights from {model_checkpoint if model_checkpoint else checkpointer.get_checkpoint_file()}"
    )

    model.post_init()
    model.to(global_torch_device())

    image_paths = list(images_dir.iterdir())

    cpu_device = torch.device("cpu")
    transforms = SSDTransform(
        cfg.input.image_size, cfg.input.pixel_mean, split=SplitEnum.testing
    )
    model.eval()

    for i, image_path in enumerate(image_paths):
        start = time.time()
        image_name = os.path.basename(image_path)

        image = numpy.array(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        images = transforms(image)[0].unsqueeze(0)
        load_time = time.time() - start

        start = time.time()
        result = model(images.to(global_torch_device()))[0]
        inference_time = time.time() - start

        result.boxes[:, 0::2] *= width / result.img_width
        result.boxes[:, 1::2] *= height / result.img_height
        (boxes, labels, scores) = (
            result.boxes.to(cpu_device).numpy(),
            result.labels.to(cpu_device).numpy(),
            result.scores.to(cpu_device).numpy(),
        )

        indices = scores > score_threshold
        boxes, labels, scores = boxes[indices], labels[indices], scores[indices]
        meters = " | ".join(
            [
                f"objects {len(boxes):02d}",
                f"load {round(load_time * 1000):03d}ms",
                f"inference {round(inference_time * 1000):03d}ms",
                f"FPS {round(1.0 / inference_time)}",
            ]
        )
        print(f"({i + 1:04d}/{len(image_paths):04d}) {image_name}: {meters}")

        drawn_image = draw_bounding_boxes(
            image,
            boxes,
            labels=labels,
            scores=scores,
            categories=categories,
            score_font=ImageFont.truetype(
                PACKAGE_DATA_PATH / "Lato-Regular.ttf",
                24,
            ),
        ).astype(numpy.uint8)
        Image.fromarray(drawn_image).save(str(output_dir / image_name))


def main():
    """description"""
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
        help="Trained " "weights.",
    )
    parser.add_argument("--score_threshold", type=float, default=0.7)
    parser.add_argument(
        "--images_dir",
        default=Path.home() / "Data" / "Neodroid",
        type=str,
        help="Specify a image dir to do prediction.",
    )

    args = parser.parse_args()

    run_demo(
        cfg=base_cfg,
        categories=base_cfg.dataset_type.categories,
        model_checkpoint=args.ckpt,
        score_threshold=args.score_threshold,
        images_dir=Path(args.images_dir),
        output_dir=Path(base_cfg.OUTPUT_DIR),
    )


if __name__ == "__main__":
    main()
