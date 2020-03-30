import argparse
import datetime
import logging
import os
import time
from pathlib import Path

import torch

from draugr.torch_utilities.optimisation.lr_scheduler import WarmupMultiStepLR
from draugr.torch_utilities.sessions import TorchCacheSession
from neodroidvision import PROJECT_APP_PATH
from neodroidvision.detection.single_stage.ssd.architecture import SingleShotDectection
from neodroidvision.detection.single_stage.ssd.evaluation import do_ssd_evaluation
from neodroidvision.detection.single_stage.ssd.metrics import (
    reduce_loss_dict,
    write_metrics_recursive,
)
from neodroidvision.detection.single_stage.ssd.multi_box_loss import MultiBoxLoss
from neodroidvision.detection.single_stage.ssd.object_detection_dataloader import (
    object_detection_data_loaders,
)
from neodroidvision.utilities import MetricLogger
from neodroidvision.data.datasets.supervised.splitting import Split
from neodroidvision.utilities.torch_utilities.check_pointer import CheckPointer
from neodroidvision.utilities.torch_utilities.distributing.distributing_utilities import (
    global_distribution_rank,
    set_benchmark_device_dist,
    setup_distributed_logger,
)
from warg import NOD
from warg.arguments import str2bool


def inner_train_ssd(
    data_root: Path,
    cfg,
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    arguments,
    kws: NOD,
):
    """

  :param data_root:
  :type data_root:
  :param cfg:
  :type cfg:
  :param model:
  :type model:
  :param data_loader:
  :type data_loader:
  :param optimizer:
  :type optimizer:
  :param scheduler:
  :type scheduler:
  :param checkpointer:
  :type checkpointer:
  :param device:
  :type device:
  :param arguments:
  :type arguments:
  :param kws:
  :type kws:
  :return:
  :rtype:
  """
    logger = logging.getLogger("SSD.trainer")
    logger.info("Start training ...")
    meters = MetricLogger()

    model.train()
    save_to_disk = global_distribution_rank() == 0
    if kws.use_tensorboard and save_to_disk:
        import tensorboardX

        summary_writer = tensorboardX.SummaryWriter(
            log_dir=str(PROJECT_APP_PATH.user_data / "results" / "tf_logs")
        )
    else:
        summary_writer = None

    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        targets = targets.to(device)
        loss_instance = MultiBoxLoss(neg_pos_ratio=cfg.MODEL.NEG_POS_RATIO)
        cls_logits, bbox_pred = model(images)

        reg_loss, cls_loss = loss_instance(
            cls_logits, bbox_pred, targets["labels"], targets["boxes"]
        )
        loss_dict = dict(reg_loss=reg_loss, cls_loss=cls_loss)

        loss = sum(loss for loss in loss_dict.values())

        loss_dict_reduced = reduce_loss_dict(
            loss_dict
        )  # reduce losses over all GPUs for logging purposes
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(total_loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time)
        if iteration % kws.log_step == 0:
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                meters.delimiter.join(
                    [
                        f"iter: {iteration:06d}",
                        f"lr: {optimizer.param_groups[0]['lr']:.5f}",
                        f"{str(meters)}",
                        f"eta: {eta_string}",
                        f"mem: {round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)}M",
                    ]
                )
            )
            if summary_writer:
                global_step = iteration
                summary_writer.add_scalar(
                    "losses/total_loss", losses_reduced, global_step=global_step
                )
                for loss_name, loss_item in loss_dict_reduced.items():
                    summary_writer.add_scalar(
                        f"losses/{loss_name}", loss_item, global_step=global_step
                    )
                summary_writer.add_scalar(
                    "lr", optimizer.param_groups[0]["lr"], global_step=global_step
                )

        if iteration % kws.save_step == 0:
            checkpointer.save(f"model_{iteration:06d}", **arguments)

        if (
            kws.eval_step > 0
            and iteration % kws.eval_step == 0
            and not iteration == max_iter
        ):
            eval_results = do_ssd_evaluation(
                data_root, cfg, model, distributed=kws.distributed, iteration=iteration
            )
            if global_distribution_rank() == 0 and summary_writer:
                for eval_result, dataset in zip(eval_results, cfg.DATASETS.TEST):
                    write_metrics_recursive(
                        eval_result["metrics"],
                        "metrics/" + dataset,
                        summary_writer,
                        iteration,
                    )
            model.train()  # *IMPORTANT*: change to train mode after eval.

    checkpointer.save("model_final", **arguments)
    # compute training time
    total_training_time = int(time.time() - start_training_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        f"Total training time: {total_time_str} ({total_training_time / max_iter:.4f} s / it)"
    )
    return model


def train_ssd(data_root: Path, cfg, kws: NOD):
    logger = logging.getLogger("SSD.trainer")
    model = SingleShotDectection(cfg)
    device = torch.device(cfg.MODEL.DEVICE)

    if kws.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[kws.local_rank], output_device=kws.local_rank
        )

    lr = cfg.SOLVER.LR * kws.num_gpus  # scale by num gpus
    lr = cfg.SOLVER.BASE_LR if lr is None else lr
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=cfg.SOLVER.MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    milestones = [step // kws.num_gpus for step in cfg.SOLVER.LR_STEPS]
    scheduler = WarmupMultiStepLR(
        optimizer=optimizer,
        milestones=cfg.SOLVER.LR_STEPS if milestones is None else milestones,
        gamma=cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
    )

    arguments = {"iteration": 0}
    save_to_disk = global_distribution_rank() == 0
    checkpointer = CheckPointer(
        model, optimizer, scheduler, cfg.OUTPUT_DIR, save_to_disk, logger
    )
    arguments.update(checkpointer.load())

    model.post_init()
    model.to(device)

    model = inner_train_ssd(
        data_root,
        cfg,
        model,
        object_detection_data_loaders(
            data_root,
            cfg,
            split=Split.Training,
            distributed=kws.distributed,
            max_iter=cfg.SOLVER.MAX_ITER // kws.num_gpus,
            start_iter=arguments["iteration"],
        ),
        optimizer,
        scheduler,
        checkpointer,
        device,
        arguments,
        kws,
    )
    return model


def main():
    from neodroidvision.detection.single_stage.ssd.config.base_config import base_cfg

    parser = argparse.ArgumentParser(
        description="Single Shot MultiBox Detector Training With PyTorch"
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
    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    args.num_gpus = num_gpus

    set_benchmark_device_dist(args.distributed, args.local_rank)
    logger = setup_distributed_logger(
        "SSD", global_distribution_rank(), PROJECT_APP_PATH.user_data / "results"
    )
    logger.info(f"Using {num_gpus} GPUs")
    logger.info(args)
    with TorchCacheSession():
        model = train_ssd(base_cfg.DATA_DIR, base_cfg, NOD(**args.__dict__))

    if not args.skip_test:
        logger.info("Start evaluating...")
        do_ssd_evaluation(base_cfg, model, distributed=args.distributed)


if __name__ == "__main__":
    main()
