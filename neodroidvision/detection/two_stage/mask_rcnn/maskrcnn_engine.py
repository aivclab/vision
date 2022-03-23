import math
import sys
import time

import torch
import tqdm
from draugr.torch_utilities import (
    TorchEvalSession,
    TorchTrainSession,
    global_torch_device,
)
from draugr.writers import Writer
from torch.nn import Module
from torch.utils.data import DataLoader

from neodroidvision.data.detection.coco import (
    CocoEvaluator,
    get_coco_api_from_dataset,
    get_iou_types,
)
from neodroidvision.utilities import reduce_dict

__all__ = ["maskrcnn_train_single_epoch", "maskrcnn_evaluate"]


def maskrcnn_train_single_epoch(
    *,
    model: Module,
    optimiser: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device = global_torch_device(),
    writer: Writer = None,
) -> None:
    """

    :param model:
    :param optimiser:
    :param data_loader:
    :param epoch_i:
    :param log_frequency:
    :param device:
    :param writer:
    :return:
    """
    model.to(device)
    with TorchTrainSession(model):

        for images, targets in tqdm.tqdm(data_loader, desc="Batch #"):
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

            optimiser.zero_grad()
            losses.backward()
            optimiser.step()

            if writer:
                for k, v in {
                    "loss": losses_reduced,
                    "lr": torch.optim.Optimizer.param_groups[0]["lr"],
                    **loss_dict_reduced,
                }.items():
                    writer.scalar(k, v)


def maskrcnn_evaluate(
    model: Module,
    data_loader: DataLoader,
    *,
    device=global_torch_device(),
    writer: Writer = None,
) -> CocoEvaluator:
    """

    Args:
      model:
      data_loader:
      device:
      writer:

    Returns:

    """
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    coco_evaluator = CocoEvaluator(
        get_coco_api_from_dataset(data_loader.dataset), get_iou_types(model)
    )

    with torch.no_grad():
        with TorchEvalSession(model):

            for image, targets in tqdm.tqdm(data_loader):
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
                if writer:
                    writer.scalar("model_time", model_time)
                    writer.scalar("evaluator_time", evaluator_time)

            coco_evaluator.synchronize_between_processes()
            coco_evaluator.accumulate()
            coco_evaluator.summarize()

    torch.set_num_threads(n_threads)

    return coco_evaluator
