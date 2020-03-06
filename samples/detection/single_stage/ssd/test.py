import argparse
import logging
import os

import torch
import torch.utils.data

from neodroidvision import PROJECT_APP_PATH
from neodroidvision.detection.single_stage.ssd.config.base_config import base_cfg
from neodroidvision.detection.single_stage.ssd.engine.inference import do_evaluation
from neodroidvision.detection.single_stage.ssd.ssd_utilities import (
    setup_logger,
    synchronize,
    CheckPointer,
)
from neodroidvision.utilities.misc.exclude import dist_util


def evaluation(cfg, ckpt, distributed):
    logger = logging.getLogger("SSD.inference")

    model = cfg.MODEL.META_ARCHITECTURE(cfg)
    checkpointer = CheckPointer(
        model, save_dir=PROJECT_APP_PATH.user_data / "results", logger=logger
    )
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    do_evaluation(cfg, model, distributed)


def main():
    parser = argparse.ArgumentParser(
        description="SSD Evaluation on VOC and COCO dataset."
    )
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
        type=str,
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    base_cfg.merge_from_file(args.config_file)
    base_cfg.merge_from_list(args.opts)
    base_cfg.freeze()

    logger = setup_logger(
        "SSD", dist_util.get_rank(), PROJECT_APP_PATH.user_data / "results"
    )
    logger.info(f"Using {num_gpus} GPUs")
    logger.info(args)

    evaluation(base_cfg, ckpt=args.ckpt, distributed=distributed)


if __name__ == "__main__":
    main()
