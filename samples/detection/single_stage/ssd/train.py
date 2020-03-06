import argparse
import logging
import os

import torch
import torch.distributed as dist

from neodroidvision import PROJECT_APP_PATH
from neodroidvision.utilities.data.datasets import make_data_loader
from neodroidvision.detection.single_stage.ssd.config.base_config import base_cfg
from neodroidvision.detection.single_stage.ssd.engine.inference import do_evaluation
from neodroidvision.detection.single_stage.ssd.engine.trainer import do_train
from neodroidvision.detection.single_stage.ssd.ssd_utilities import (
    CheckPointer,
    make_lr_scheduler,
    make_optimizer,
    setup_logger,
    synchronize,
)
from neodroidvision.utilities.misc.exclude import dist_util
from warg.arguments import str2bool


def train(cfg, args):
    logger = logging.getLogger("SSD.trainer")
    model = cfg.MODEL.META_ARCHITECTURE(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )

    lr = cfg.SOLVER.LR * args.num_gpus  # scale by num gpus
    optimizer = make_optimizer(cfg, model, lr)

    milestones = [step // args.num_gpus for step in cfg.SOLVER.LR_STEPS]
    scheduler = make_lr_scheduler(cfg, optimizer, milestones)

    arguments = {"iteration": 0}
    save_to_disk = dist_util.get_rank() == 0
    checkpointer = CheckPointer(
        model, optimizer, scheduler, cfg.OUTPUT_DIR, save_to_disk, logger
    )
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)

    max_iter = cfg.SOLVER.MAX_ITER // args.num_gpus
    train_loader = make_data_loader(
        cfg,
        is_train=True,
        distributed=args.distributed,
        max_iter=max_iter,
        start_iter=arguments["iteration"],
    )

    model = do_train(
        cfg,
        model,
        train_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        arguments,
        args,
    )
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Single Shot MultiBox Detector Training With PyTorch"
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
        "--log_step", default=10, type=int, help="Print logs every log_step"
    )
    parser.add_argument(
        "--save_step", default=2500, type=int, help="Save checkpoint every save_step"
    )
    parser.add_argument(
        "--eval_step",
        default=2500,
        type=int,
        help="Evaluate dataset every eval_step, disabled when eval_step < 0",
    )
    parser.add_argument("--use_tensorboard", default=True, type=str2bool)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    args.num_gpus = num_gpus

    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True
    if args.distributed:
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

    model = train(base_cfg, args)

    if not args.skip_test:
        logger.info("Start evaluating...")
        torch.cuda.empty_cache()  # speed up evaluating after training finished
        do_evaluation(base_cfg, model, distributed=args.distributed)


if __name__ == "__main__":
    main()
