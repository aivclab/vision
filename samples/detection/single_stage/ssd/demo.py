import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from draugr import global_torch_device
from draugr.python_utilities.paths import ensure_existence
from neodroidvision import PROJECT_APP_PATH
from neodroidvision.detection.single_stage.ssd.ssd_utilities import (
    CheckPointer,
    draw_boxes,
)
from neodroidvision.detection.single_stage.ssd.engine.factories.transforms_factory import (
    build_transforms,
)
from neodroidvision.utilities.data.datasets.supervised.detection import (
    COCODataset,
    VOCDataset,
)


@torch.no_grad()
def run_demo(cfg, model_ckpt, score_threshold, images_dir, output_dir, dataset_type):
    if dataset_type == "voc":
        class_names = VOCDataset.class_names
    elif dataset_type == "coco":
        class_names = COCODataset.class_names
    else:
        raise NotImplementedError(dataset_type)

    model = cfg.MODEL.META_ARCHITECTURE(cfg)
    model = model.to(global_torch_device())
    save_dir = PROJECT_APP_PATH.user_data / "results"
    ensure_existence(save_dir)
    checkpointer = CheckPointer(model, save_dir=save_dir)
    checkpointer.load(model_ckpt, use_latest=model_ckpt is None)
    weight_file = model_ckpt if model_ckpt else checkpointer.get_checkpoint_file()
    print(f"Loaded weights from {weight_file}")

    image_paths = list(images_dir.iterdir())

    cpu_device = torch.device("cpu")
    transforms = build_transforms(cfg, is_train=False)
    model.eval()
    for i, image_path in enumerate(image_paths):
        start = time.time()
        image_name = os.path.basename(image_path)

        image = np.array(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        images = transforms(image)[0].unsqueeze(0)
        load_time = time.time() - start

        start = time.time()
        result = model(images.to(global_torch_device()))[0]
        inference_time = time.time() - start

        result = result.resize((width, height)).to(cpu_device).numpy()
        boxes, labels, scores = result["boxes"], result["labels"], result["scores"]

        indices = scores > score_threshold
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
        meters = " | ".join(
            [
                f"objects {len(boxes):02d}",
                f"load {round(load_time * 1000):03d}ms",
                f"inference {round(inference_time * 1000):03d}ms",
                f"FPS {round(1.0 / inference_time)}",
            ]
        )
        print(f"({i + 1:04d}/{len(image_paths):04d}) {image_name}: {meters}")

        drawn_image = draw_boxes(image, boxes, labels, scores, class_names).astype(
            np.uint8
        )
        Image.fromarray(drawn_image).save(os.path.join(output_dir, image_name))


def main():
    from configs.vgg_ssd300_coco_trainval35k import base_cfg

    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/heider/Projects/Alexandra/Python/vision/samples/detection/single_stage/ssd/exclude/models/vgg_ssd300_coco_trainval35k.pth",
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
        model_ckpt=args.ckpt,
        score_threshold=args.score_threshold,
        images_dir=Path(args.images_dir),
        output_dir=base_cfg.OUTPUT_DIR,
        dataset_type=base_cfg.dataset_type,
    )


if __name__ == "__main__":
    main()
