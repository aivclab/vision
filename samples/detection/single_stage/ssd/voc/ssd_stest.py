import argparse
import logging
import os

import torch
import torch.utils.data
from draugr.torch_utilities import global_torch_device
from draugr.torch_utilities.sessions import TorchCacheSession
from utilities.torch_utilities.persistence.check_pointer import CheckPointer

from neodroidvision import PROJECT_APP_PATH
from neodroidvision.detection.single_stage.ssd.architecture import SingleShotDetection
from neodroidvision.detection.single_stage.ssd.ssd_evaluation import do_ssd_evaluation
from neodroidvision.utilities.torch_utilities.distributing.distributing_utilities import (
    global_distribution_rank,
    set_benchmark_device_dist,
    setup_distributed_logger,
)


def main():
    """ """
    from configs.vgg_ssd300_coco_trainval35k import base_cfg

    parser = argparse.ArgumentParser(
        description="SSD Evaluation on VOC and COCO dataset."
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=PROJECT_APP_PATH.user_data
        / "ssd"
        / "models"
        / "mobilenet_v2_ssd320_voc0712.pth"
        # "mobilenet_v2_ssd320_voc0712.pth"
        # "vgg_ssd300_coco_trainval35k.pth"
        # "vgg_ssd512_coco_trainval35k.pth"
        ,
        type=str,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    set_benchmark_device_dist(distributed, args.local_rank)

    logger = setup_distributed_logger(
        "SSD", global_distribution_rank(), PROJECT_APP_PATH.user_data / "results"
    )
    logger.info(f"Using {num_gpus} GPUs")
    logger.info(args)

    device = torch.device(base_cfg.MODEL.DEVICE)
    global_torch_device(override=device)

    with TorchCacheSession():
        model = SingleShotDetection(base_cfg)
        checkpointer = CheckPointer(
            model,
            save_dir=PROJECT_APP_PATH.user_data / "results",
            logger=logging.getLogger("SSD.inference"),
        )

        checkpointer.load(args.ckpt, use_latest=args.ckpt is None)
        do_ssd_evaluation(
            base_cfg.data_dir,
            base_cfg,
            model.to(torch.device(base_cfg.MODEL.DEVICE)),
            distributed,
        )


if __name__ == "__main__":
    main()
