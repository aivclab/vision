import argparse
from pathlib import Path
from typing import List

import cv2
import numpy
import torch
from tqdm import tqdm

from apppath import ensure_existence
from draugr.opencv_utilities import draw_bouding_boxes, frame_generator
from draugr.torch_utilities import TorchEvalSession, global_torch_device
from neodroidvision import PROJECT_APP_PATH
from neodroidvision.data.datasets.supervised.splitting import Split
from neodroidvision.detection import SingleShotDectection
from neodroidvision.detection.single_stage.ssd.bounding_boxes.ssd_transforms import (
    SSDTransform,
)
from neodroidvision.utilities import CheckPointer
from warg import NOD


@torch.no_grad()
def run_webcam_demo(
    cfg: NOD,
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
        cfg.INPUT.IMAGE_SIZE, cfg.INPUT.PIXEL_MEAN, split=Split.Testing
    )
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

    with TorchEvalSession(model):
        for image in tqdm(frame_generator(cv2.VideoCapture(0))):
            result = model(transforms(image)[0].unsqueeze(0).to(global_torch_device()))[
                0
            ]
            height, width, *_ = image.shape

            result["boxes"][:, 0::2] *= width / result["img_width"]
            result["boxes"][:, 1::2] *= height / result["img_height"]
            (boxes, labels, scores) = (
                result["boxes"].to(cpu_device).numpy(),
                result["labels"].to(cpu_device).numpy(),
                result["scores"].to(cpu_device).numpy(),
            )

            indices = scores > score_threshold

            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(
                window_name,
                draw_bouding_boxes(
                    image, boxes[indices], labels[indices], scores[indices], categories
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
        default="/home/heider/Projects/Alexandra/Python/vision/samples/detection/single_stage"
        "/ssd/exclude"
        "/models/vgg_ssd300_coco_trainval35k.pth",
        help="Use weights from path",
    )
    parser.add_argument("--score_threshold", type=float, default=0.7)
    args = parser.parse_args()

    run_webcam_demo(
        cfg=base_cfg,
        categories=base_cfg.dataset_type.categories,
        model_ckpt=Path(args.ckpt),
        score_threshold=args.score_threshold,
    )


if __name__ == "__main__":
    main()
