import os

from torch.utils.data import ConcatDataset

from neodroidvision.utilities.data import COCODataset, coco_evaluation
from neodroidvision.utilities.data import VOCDataset, voc_evaluation

_DATASETS = {"VOCDataset": VOCDataset, "COCODataset": COCODataset}


class DatasetCatalog:
    DATA_DIR = "datasets"
    DATASETS = {
        "voc_2007_train": {"data_dir": "VOC2007", "split": "train"},
        "voc_2007_val": {"data_dir": "VOC2007", "split": "val"},
        "voc_2007_trainval": {"data_dir": "VOC2007", "split": "trainval"},
        "voc_2007_test": {"data_dir": "VOC2007", "split": "test"},
        "voc_2012_train": {"data_dir": "VOC2012", "split": "train"},
        "voc_2012_val": {"data_dir": "VOC2012", "split": "val"},
        "voc_2012_trainval": {"data_dir": "VOC2012", "split": "trainval"},
        "voc_2012_test": {"data_dir": "VOC2012", "split": "test"},
        "coco_2014_valminusminival": {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_valminusminival2014.json",
        },
        "coco_2014_minival": {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_minival2014.json",
        },
        "coco_2014_train": {
            "data_dir": "train2014",
            "ann_file": "annotations/instances_train2014.json",
        },
        "coco_2014_val": {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_val2014.json",
        },
    }

    @staticmethod
    def get(name):
        if "voc" in name:
            voc_root = DatasetCatalog.DATA_DIR
            if "VOC_ROOT" in os.environ:
                voc_root = os.environ["VOC_ROOT"]

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(voc_root, attrs["data_dir"]), split=attrs["split"]
            )
            return dict(factory="VOCDataset", args=args)
        elif "coco" in name:
            coco_root = DatasetCatalog.DATA_DIR
            if "COCO_ROOT" in os.environ:
                coco_root = os.environ["COCO_ROOT"]

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(coco_root, attrs["data_dir"]),
                ann_file=os.path.join(coco_root, attrs["ann_file"]),
            )
            return dict(factory="COCODataset", args=args)

        raise RuntimeError(f"Dataset not available: {name}")


def build_dataset(dataset_list, transform=None, target_transform=None, is_train=True):
    assert len(dataset_list) > 0
    datasets = []
    for dataset_name in dataset_list:
        data = DatasetCatalog.get(dataset_name)
        args = data["args"]
        factory = _DATASETS[data["factory"]]
        args["transform"] = transform
        args["target_transform"] = target_transform
        if factory == VOCDataset:
            args["keep_difficult"] = not is_train
        elif factory == COCODataset:
            args["remove_empty"] = is_train
        dataset = factory(**args)
        datasets.append(dataset)
    # for testing, return a list of datasets
    if not is_train:
        return datasets
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)

    return [dataset]


def evaluate(dataset, predictions, output_dir, **kwargs):
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
