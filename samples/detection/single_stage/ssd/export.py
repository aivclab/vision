import argparse
from pathlib import Path

import cv2
import torch
from torch import onnx
from tqdm import tqdm

from apppath import ensure_existence
from draugr.opencv_utilities.frames import frame_generator
from draugr.torch_utilities import global_torch_device
from neodroidvision import PROJECT_APP_PATH
from neodroidvision.detection.single_stage.ssd.architecture import SingleShotDectection
from neodroidvision.data.datasets.supervised.splitting import Split
from neodroidvision.detection.single_stage.ssd.bounding_boxes.ssd_transforms import (
    SSDTransform,
)
from neodroidvision.utilities.torch_utilities.check_pointer import CheckPointer
from warg import NOD


@torch.no_grad()
def export_m(
    cfg: NOD,
    model_ckpt: Path,
    model_onnx_path: Path = Path("torch_model.onnx"),
    verbose: bool = True,
):
    """

  :param cfg:
  :type cfg:
  :param model_ckpt:
  :type model_ckpt:
  :param model_onnx_path:
  :type model_onnx_path:
  :return:
  :rtype:
  """
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

    transforms = SSDTransform(
        cfg.INPUT.IMAGE_SIZE, cfg.INPUT.PIXEL_MEAN, split=Split.Testing
    )
    model.eval()

    for image in tqdm(frame_generator(cv2.VideoCapture(0))):
        example_input = (transforms(image)[0].unsqueeze(0).to(global_torch_device()),)
        output = onnx.export(
            model, example_input, str(model_onnx_path), verbose=verbose
        )


def main():
    from configs.vgg_ssd300_coco_trainval35k import base_cfg

    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/heider/Projects/Alexandra/Python/vision/samples/detection/single_stage"
        "/ssd/exclude"
        "/models/vgg_ssd300_coco_trainval35k.pth",
        help="Trained " "weights.",
    )
    args = parser.parse_args()

    export_m(cfg=base_cfg, model_ckpt=Path(args.ckpt))


if __name__ == "__main__":
    main()
