from data.detection.voc import VOCDataset
from neodroidvision.detection import SSDLiteBoxPredictor
from neodroidvision.detection.single_stage.ssd.architecture.backbones import (
    mobilenet_v2_factory,
)
from neodroidvision.detection.single_stage.ssd.config.ssd_base_config import base_cfg

base_cfg.data_dir = base_cfg.data_dir / "PASCAL" / "Train"

base_cfg.model.backbone.update(
    name=mobilenet_v2_factory,
    out_channels=(96, 1280, 512, 256, 256, 64),
    predictor_type=SSDLiteBoxPredictor,
),
base_cfg.model.box_head.priors.update(
    feature_maps=(20, 10, 5, 3, 2, 1),
    strides=(16, 32, 64, 106, 160, 320),
    min_sizes=(60, 105, 150, 195, 240, 285),
    max_sizes=(105, 150, 195, 240, 285, 330),
    aspect_ratios=((2, 3), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3)),
    boxes_per_location=(6, 6, 6, 6, 6, 6),
)
base_cfg.input.update(image_size=320)

base_cfg.dataset_type = VOCDataset
base_cfg.datasets.update(
    train=("voc_2007_trainval", "voc_2012_trainval"), test=("voc_2007_test",)
)
base_cfg.solver.update(
    max_iter=120000, lr_steps=(80000, 100000), gamma=0.1, batch_size=32, lr=1e-3
)
base_cfg.model.box_head.update(num_categories=len(base_cfg.dataset_type.categories))
