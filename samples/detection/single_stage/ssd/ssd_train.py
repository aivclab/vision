import argparse
import datetime
import logging
import os
import time
from pathlib import Path

import torch
from apppath import ensure_existence
from draugr.numpy_utilities import Split

from neodroidvision import PROJECT_APP_PATH
from neodroidvision.detection.single_stage.ssd import (
    MultiBoxLoss,
    SingleShotDetectionNms,
    do_ssd_evaluation,
    object_detection_data_loaders,
)
from neodroidvision.utilities import (
    CheckPointer,
    MetricLogger,
    global_distribution_rank,
    reduce_loss_dict,
    set_benchmark_device_dist,
    setup_distributed_logger,
    write_metrics_recursive,
)
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from warg import NOD
from warg.arguments import str2bool

from draugr.torch_utilities import (
    TorchCacheSession,
    TorchEvalSession,
    TorchTrainSession,
    WarmupMultiStepLR,
)


def inner_train_ssd(
    *,
    data_root: Path,
    cfg: NOD,
    model: Module,
    data_loader: DataLoader,
    optimiser: Optimizer,
    scheduler: WarmupMultiStepLR,
    check_pointer: callable,
    device: callable,
    arguments: callable,
    kws: NOD,
) -> Module:
    """

    :param data_root:
    :type data_root:
    :param cfg:
    :type cfg:
    :param model:
    :type model:
    :param data_loader:
    :type data_loader:
    :param optimiser:
    :type optimiser:
    :param scheduler:
    :type scheduler:
    :param check_pointer:
    :type check_pointer:
    :param device:
    :type device:
    :param arguments:
    :type arguments:
    :param kws:
    :type kws:
    :return:
    :rtype:"""
    logger = logging.getLogger("SSD.trainer")
    logger.info("Start training ...")
    meters = MetricLogger()

    with TorchTrainSession(model):
        save_to_disk = global_distribution_rank() == 0
        if kws.use_tensorboard and save_to_disk:
            import tensorboardX

            writer = tensorboardX.SummaryWriter(
                log_dir=str(PROJECT_APP_PATH.user_data / "results" / "tf_logs")
            )
        else:
            writer = None

        max_iter = len(data_loader)
        start_iter = arguments["iteration"]
        start_training_time = time.time()
        end = time.time()
        for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
            arguments["iteration"] = iteration

            images = images.to(device)
            targets = targets.to(device)
            loss_instance = MultiBoxLoss(neg_pos_ratio=cfg.model.neg_pos_ratio)
            cls_logits, bbox_pred = model(images)

            reg_loss, cls_loss = loss_instance(
                cls_logits, bbox_pred, targets.labels, targets.boxes
            )
            loss_dict = dict(reg_loss=reg_loss, cls_loss=cls_loss)

            loss = sum(loss for loss in loss_dict.values())

            loss_dict_reduced = reduce_loss_dict(
                loss_dict
            )  # reduce losses over all GPUs for logging purposes
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(total_loss=losses_reduced, **loss_dict_reduced)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
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
                            f"lr: {optimiser.param_groups[0]['lr']:.5f}",
                            f"{str(meters)}",
                            f"eta: {eta_string}",
                            f"mem: {round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)}M",
                        ]
                    )
                )
                if writer:
                    global_step = iteration
                    writer.add_scalar(
                        "losses/total_loss", losses_reduced, global_step=global_step
                    )
                    for loss_name, loss_item in loss_dict_reduced.items():
                        writer.add_scalar(
                            f"losses/{loss_name}", loss_item, global_step=global_step
                        )
                    writer.add_scalar(
                        "lr", optimiser.param_groups[0]["lr"], global_step=global_step
                    )

            if iteration % kws.save_step == 0:
                check_pointer.save(f"model_{iteration:06d}", **arguments)

            if (
                kws.eval_step > 0
                and iteration % kws.eval_step == 0
                and not iteration == max_iter
            ):
                with TorchEvalSession(model):
                    eval_results = do_ssd_evaluation(
                        data_root,
                        cfg,
                        model,
                        distributed=kws.distributed,
                        iteration=iteration,
                    )
                    if global_distribution_rank() == 0 and writer:
                        for eval_result, dataset in zip(
                            eval_results, cfg.datasets.test
                        ):
                            write_metrics_recursive(
                                eval_result["metrics"],
                                "metrics/" + dataset,
                                writer,
                                iteration,
                            )

        check_pointer.save("model_final", **arguments)

        total_training_time = int(
            time.time() - start_training_time
        )  # compute training time
        logger.info(
            f"Total training time: {datetime.timedelta(seconds=total_training_time)} ("
            f"{total_training_time / max_iter:.4f} s / it)"
        )
        return model


def train_ssd(data_root: Path, cfg, solver_cfg: NOD, kws: NOD) -> Module:
    logger = logging.getLogger("SSD.trainer")
    model = SingleShotDetectionNms(cfg)
    device = torch.device(cfg.model.device)

    if kws.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[kws.local_rank], output_device=kws.local_rank
        )

    lr = solver_cfg.lr * kws.num_gpus  # scale by num gpus
    lr = solver_cfg.base_lr if lr is None else lr
    optimiser = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=solver_cfg.momentum,
        weight_decay=solver_cfg.weight_decay,
    )

    milestones = [step // kws.num_gpus for step in solver_cfg.lr_steps]
    scheduler = WarmupMultiStepLR(
        optimiser=optimiser,
        milestones=solver_cfg.lr_steps if milestones is None else milestones,
        gamma=solver_cfg.gamma,
        warmup_factor=solver_cfg.warmup_factor,
        warmup_iters=solver_cfg.warmup_iters,
    )

    arguments = {"iteration": 0}
    save_to_disk = global_distribution_rank() == 0
    checkpointer = CheckPointer(
        model, optimiser, scheduler, cfg.output_dir, save_to_disk, logger
    )
    arguments.update(checkpointer.load())

    model.post_init()
    model.to(device)

    model = inner_train_ssd(
        data_root=data_root,
        cfg=cfg,
        model=model,
        data_loader=object_detection_data_loaders(
            data_root=data_root,
            cfg=cfg,
            split=Split.Training,
            distributed=kws.distributed,
            max_iter=solver_cfg.max_iter // kws.num_gpus,
            start_iter=arguments["iteration"],
        ),
        optimiser=optimiser,
        scheduler=scheduler,
        check_pointer=checkpointer,
        device=device,
        arguments=arguments,
        kws=kws,
    )
    return model


def main():
    from configs.mobilenet_v2_ssd320_voc0712 import base_cfg

    # from configs.efficient_net_b3_ssd300_voc0712 import base_cfg
    # from configs.vgg_ssd300_voc0712 import base_cfg

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
        "SSD",
        global_distribution_rank(),
        ensure_existence(PROJECT_APP_PATH.user_data / "results"),
    )
    logger.info(f"Using {num_gpus} GPUs")
    logger.info(args)
    with TorchCacheSession():
        model = train_ssd(
            base_cfg.data_dir, base_cfg, base_cfg.solver, NOD(**args.__dict__)
        )

    if not args.skip_test:
        logger.info("Start evaluating...")
        do_ssd_evaluation(base_cfg, model, distributed=args.distributed)


if __name__ == "__main__":
    main()
