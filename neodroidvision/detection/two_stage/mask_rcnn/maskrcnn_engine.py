import math
import sys
import time

import torch
import torchvision
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from draugr.torch_utilities import (
    TorchEvalSession,
    TorchTrainSession,
    global_torch_device,
    warmup_lr_scheduler,
)
from neodroidvision.data.detection.coco import (
    CocoEvaluator,
    get_coco_api_from_dataset,
    get_iou_types,
)
from neodroidvision.utilities import MetricLogger, SmoothedValue, reduce_dict


def train_single_epoch(
    *,
    model: torchvision.models.detection.mask_rcnn.MaskRCNN,
    optimizer: Optimizer,
    data_loader: DataLoader,
    epoch_i: int,
    log_frequency: int,
    device: torch.device = global_torch_device(),
):
    model.to(device)
    with TorchTrainSession(model):
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

        lr_scheduler = None
        if epoch_i == 0:
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(data_loader) - 1)

            lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

        for images, targets in metric_logger.log_every(
            data_loader, log_frequency, f"Epoch: [{epoch_i}]"
        ):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # torch.cuda.synchronize(device)
            loss_dict = model(images, targets=targets)

            losses = sum(loss for loss in loss_dict.values())

            loss_dict_reduced = reduce_dict(
                loss_dict
            )  # reduce losses over all GPUs for logging purposes
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def maskrcnn_evaluate(
    model: Module, data_loader: DataLoader, device=global_torch_device()
):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")

    with torch.no_grad():
        with TorchEvalSession(model):

            metric_logger = MetricLogger(delimiter="  ")
            coco_evaluator = CocoEvaluator(
                get_coco_api_from_dataset(data_loader.dataset), get_iou_types(model)
            )

            for image, targets in metric_logger.log_every(data_loader, 100, "Test:"):
                image = [img.to(device) for img in image]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                torch.cuda.synchronize(device)
                model_time = time.time()
                outputs = model(image)

                outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
                model_time = time.time() - model_time

                res = {
                    target["image_id"].item(): output
                    for target, output in zip(targets, outputs)
                }
                evaluator_time = time.time()
                coco_evaluator.update(res)
                evaluator_time = time.time() - evaluator_time
                metric_logger.update(
                    model_time=model_time, evaluator_time=evaluator_time
                )

            # gather the stats from all processes
            metric_logger.synchronise_meters_between_processes()
            print("Averaged stats:", metric_logger)
            coco_evaluator.synchronize_between_processes_aaa()

            # accumulate predictions from all images
            coco_evaluator.accumulate()
            coco_evaluator.summarize()

    torch.set_num_threads(n_threads)

    return coco_evaluator
