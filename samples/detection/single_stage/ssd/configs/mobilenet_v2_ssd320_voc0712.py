from neodroidvision.detection.single_stage.ssd.architecture.backbone.mobilenet import (
    mobilenet_v2,
)
from neodroidvision.detection.single_stage.ssd.architecture.box_head.box_predictor import (
    SSDLiteBoxPredictor,
)

from neodroidvision.detection.single_stage.ssd.config.base_config import base_cfg

base_cfg.MODEL.update(NUM_CLASSES=21)
base_cfg.MODEL.BOX_HEAD.update(PREDICTOR=SSDLiteBoxPredictor),
base_cfg.MODEL.BACKBONE.update(
    NAME=mobilenet_v2, OUT_CHANNELS=(96, 1280, 512, 256, 256, 64)
),
base_cfg.MODEL.PRIORS.update(
    FEATURE_MAPS=[20, 10, 5, 3, 2, 1],
    STRIDES=[16, 32, 64, 100, 150, 300],
    MIN_SIZES=[60, 105, 150, 195, 240, 285],
    MAX_SIZES=[105, 150, 195, 240, 285, 330],
    ASPECT_RATIOS=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    BOXES_PER_LOCATION=[6, 6, 6, 6, 6, 6],
)
base_cfg.INPUT.update(IMAGE_SIZE=320)

base_cfg.dataset_type = "voc"
base_cfg.DATASETS.update(
    TRAIN=("voc_2007_trainval", "voc_2012_trainval"), TEST=("voc_2007_test",)
)
base_cfg.SOLVER.update(
    MAX_ITER=120000, LR_STEPS=[80000, 100000], GAMMA=0.1, BATCH_SIZE=32, LR=1e-3
)
