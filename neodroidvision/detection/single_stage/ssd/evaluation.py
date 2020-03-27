import logging
from pathlib import Path

import torch
import torch.utils.data
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from neodroidvision import PROJECT_APP_PATH
from neodroidvision.data.datasets import (
    COCODataset,
    VOCDataset,
    coco_evaluation,
    voc_evaluation,
)
from neodroidvision.detection.single_stage.ssd.object_detection_dataloader import (
    object_detection_data_loaders,
)
from neodroidvision.data.datasets.supervised.splitting import Split
from neodroidvision.utilities import (
    distributing_utilities,
    is_main_process,
    synchronise_torch_barrier,
)
from warg import NOD

__all__ = ["do_ssd_evaluation"]


def compute_on_dataset(
    model: Module, data_loader: DataLoader, device: torch.device
) -> dict:
    results_dict = {}
    for batch in tqdm(data_loader):
        images, targets, image_ids = batch
        cpu_device = torch.device("cpu")
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


def accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = distributing_utilities.all_gather(predictions_per_gpu)
    if not distributing_utilities.is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("SSD.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not a contiguous set. Some "
            "images "
            "might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def evaluate_dataset(dataset, predictions, output_dir, **kwargs):
    """evaluate dataset using different methods based on dataset type.
Args:
    dataset: Dataset object
    predictions(list[(boxes, labels, scores)]): Each item in the list represents the
        prediction results for one image. And the index should match the dataset index.
    output_dir: output folder, to save evaluation files or results.
Returns:
    evaluation result
"""
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
    model: Module,
    data_loader: DataLoader,
    dataset_name: str,
    device: torch.device,
    output_folder: Path = None,
    use_cached: bool = False,
    **kwargs,
):
    dataset = data_loader.dataset
    logger = logging.getLogger("SSD.inference")
    logger.info(f"Evaluating {dataset_name} dataset({len(dataset)} images):")

    predictions_path = output_folder / "predictions.pth"

    if use_cached and predictions_path.exists():
        predictions = torch.load(predictions_path, map_location="cpu")
    else:
        predictions = compute_on_dataset(model, data_loader, device)
        synchronise_torch_barrier()
        predictions = accumulate_predictions_from_multiple_gpus(predictions)

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
):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    model.eval()
    device = torch.device(cfg.MODEL.DEVICE)
    eval_results = []
    for dataset_name, data_loader in zip(
        cfg.DATASETS.TEST,
        object_detection_data_loaders(
            data_root, cfg, split=Split.Validation, distributed=distributed
        ),
    ):
        eval_results.append(
            inference_ssd(
                model,
                data_loader,
                dataset_name,
                device,
                PROJECT_APP_PATH.user_data / "results" / "inference" / dataset_name,
                **kwargs,
            )
        )
    return eval_results
