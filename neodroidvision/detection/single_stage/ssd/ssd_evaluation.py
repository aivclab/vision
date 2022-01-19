import logging
from pathlib import Path
from typing import Any, List

import torch
import torch.utils.data
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
from warg import NOD

from neodroidvision import PROJECT_APP_PATH
from neodroidvision.data.detection.coco import COCODataset, coco_evaluation
from neodroidvision.data.detection.voc import VOCDataset, voc_evaluation
from neodroidvision.detection.single_stage.ssd.object_detection_dataloader import (
    object_detection_data_loaders,
)
from neodroidvision.utilities import (
    distributing_utilities,
    is_main_process,
    synchronise_torch_barrier,
)

__all__ = ["do_ssd_evaluation"]

from draugr.numpy_utilities import SplitEnum


def compute_on_dataset(
    model: Module,
    data_loader: DataLoader,
    device: torch.device,
    cpu_device=torch.device("cpu"),
) -> dict:
    """

    Args:
      model:
      data_loader:
      device:
      cpu_device:

    Returns:

    """
    results_dict = {}
    for batch in tqdm(data_loader):
        images, targets, image_ids = batch
        with torch.no_grad():
            results_dict.update(
                {
                    img_id: result
                    for img_id, result in zip(
                        image_ids, [o.to(cpu_device) for o in model(images.to(device))]
                    )
                }
            )
    return results_dict


def accumulate_predictions_from_cuda_devices(predictions_per_gpu: Any) -> list:
    """

    :param predictions_per_gpu:
    :return:"""
    all_predictions = distributing_utilities.all_gather_cuda(predictions_per_gpu)
    if not distributing_utilities.is_main_process():
        return

    predictions = {}
    for p in all_predictions:  # merge the list of dicts
        predictions.update(p)

    image_ids = list(
        sorted(predictions.keys())
    )  # convert a dict where the key is the index in a list
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("SSD.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not a contiguous set. Some "
            "images "
            "might be missing from the evaluation"
        )

    return [predictions[i] for i in image_ids]


def evaluate_dataset(dataset, predictions, output_dir: Path, **kwargs) -> dict:
    """evaluate dataset using different methods based on dataset type.
    Args:
    dataset: Dataset object
    predictions(list[(boxes, labels, scores)]): Each item in the list represents the
      prediction results for one image. And the index should match the dataset index.
    output_dir: output folder, to save evaluation files or results.
    Returns:
    evaluation result"""
    kws = dict(
        dataset=dataset, predictions=predictions, output_dir=output_dir, **kwargs
    )
    if isinstance(dataset, VOCDataset):
        return voc_evaluation(**kws)
    elif isinstance(dataset, COCODataset):
        return coco_evaluation(**kws)
    else:
        raise NotImplementedError


def inference_ssd(
    *,
    model: Module,
    data_loader: DataLoader,
    dataset_name: str,
    device: torch.device,
    output_folder: Path = None,
    use_cached: bool = False,
    **kwargs,
) -> dict:
    """

    :param model:
    :param data_loader:
    :param dataset_name:
    :param device:
    :param output_folder:
    :param use_cached:
    :param kwargs:
    :return:"""
    dataset = data_loader.dataset
    logger = logging.getLogger("SSD.inference")
    logger.info(f"Evaluating {dataset_name} dataset({len(dataset)} images):")

    predictions_path = output_folder / "predictions.pth"

    if use_cached and predictions_path.exists():
        predictions = torch.load(predictions_path, map_location="cpu")
    else:
        predictions = compute_on_dataset(model, data_loader, device)
        synchronise_torch_barrier()
        predictions = accumulate_predictions_from_cuda_devices(predictions)

    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, predictions_path)

    return evaluate_dataset(
        dataset=dataset, predictions=predictions, output_dir=output_folder, **kwargs
    )


@torch.no_grad()
def do_ssd_evaluation(
    data_root: Path, cfg: NOD, model: Module, distributed: bool, **kwargs
) -> List:
    """

    Args:


    :param data_root:
    :param cfg:
    :param model:
    :param distributed:
    :param kwargs:
    :return:"""
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    model.eval()
    device = torch.device(cfg.MODEL.DEVICE)
    eval_results = []
    for dataset_name, data_loader in zip(
        cfg.DATASETS.TEST,
        object_detection_data_loaders(
            data_root=data_root,
            cfg=cfg,
            split=SplitEnum.validation,
            distributed=distributed,
        ),
    ):
        eval_results.append(
            inference_ssd(
                model=model,
                data_loader=data_loader,
                dataset_name=dataset_name,
                device=device,
                output_folder=PROJECT_APP_PATH.user_data
                / "results"
                / "inference"
                / dataset_name,
                **kwargs,
            )
        )
    return eval_results
