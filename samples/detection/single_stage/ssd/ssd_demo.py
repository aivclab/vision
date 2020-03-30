import argparse
import os
import time
from pathlib import Path

import numpy
import torch
from PIL import Image

from apppath import ensure_existence
from draugr.opencv_utilities import draw_bouding_boxes
from draugr.torch_utilities import global_torch_device
from neodroidvision import PROJECT_APP_PATH
from neodroidvision.data.datasets.supervised.splitting import Split
from neodroidvision.detection import SingleShotDectection
from neodroidvision.detection.single_stage.ssd.bounding_boxes.ssd_transforms import (
    SSDTransform,
)
from neodroidvision.utilities import CheckPointer


@torch.no_grad()
def run_demo(cfg, class_names, model_ckpt, score_threshold, images_dir, output_dir):
    model = SingleShotDectection(cfg)

    checkpointer = CheckPointer(
        model, save_dir=ensure_existence(PROJECT_APP_PATH.user_data / "results")
    )
    checkpointer.load(model_ckpt, use_latest=model_ckpt is None)
    print(
        f"Loaded weights from {model_ckpt if model_ckpt else checkpointer.get_checkpoint_file()}"
    )

    model.post_init()
    model.to(global_torch_device())

    image_paths = list(images_dir.iterdir())

    cpu_device = torch.device("cpu")
    transforms = SSDTransform(
        cfg.INPUT.IMAGE_SIZE, cfg.INPUT.PIXEL_MEAN, split=Split.Testing
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

        result["boxes"][:, 0::2] *= width / result["img_width"]
        result["boxes"][:, 1::2] *= height / result["img_height"]
        (boxes, labels, scores) = (
            result["boxes"].to(cpu_device).numpy(),
            result["labels"].to(cpu_device).numpy(),
            result["scores"].to(cpu_device).numpy(),
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

        drawn_image = draw_bouding_boxes(
            image, boxes, labels, scores, class_names
        ).astype(numpy.uint8)
        Image.fromarray(drawn_image).save(os.path.join(output_dir, image_name))


def main():
    from configs.vgg_ssd300_coco_trainval35k import base_cfg

    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/heider/Projects/Alexandra/Python/vision/samples/detection/single_stage/ssd/exclude"
        "/models/vgg_ssd300_coco_trainval35k.pth",
        help="Trained " "weights.",
    )
    parser.add_argument("--score_threshold", type=float, default=0.7)
    parser.add_argument(
        "--images_dir",
        default="/home/heider/Pictures/Neodroid/",
        type=str,
        help="Specify a image dir to do prediction.",
    )

    args = parser.parse_args()

    run_demo(
        cfg=base_cfg,
        class_names=base_cfg.dataset_type.categories,
        model_ckpt=args.ckpt,
        score_threshold=args.score_threshold,
        images_dir=Path(args.images_dir),
        output_dir=base_cfg.OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
