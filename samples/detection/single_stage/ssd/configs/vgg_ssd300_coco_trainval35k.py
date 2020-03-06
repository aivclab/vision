from neodroidvision.detection.single_stage.ssd.config.base_config import base_cfg

base_cfg.MODEL.update(NUM_CLASSES=81)
base_cfg.MODEL.PRIORS.update(
    FEATURE_MAPS=[38, 19, 10, 5, 3, 1],
    STRIDES=[8, 16, 32, 64, 100, 300],
    MIN_SIZES=[21, 45, 99, 153, 207, 261],
    MAX_SIZES=[45, 99, 153, 207, 261, 315],
    ASPECT_RATIOS=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    BOXES_PER_LOCATION=[4, 6, 6, 6, 4, 4],
)

base_cfg.INPUT.update(IMAGE_SIZE=300)

base_cfg.DATASETS.update(
    TRAIN=("coco_2014_train", "coco_2014_valminusminival"), TEST=("coco_2014_minival",)
)
base_cfg.dataset_type = "coco"
base_cfg.SOLVER.update(
    MAX_ITER=400000, LR_STEPS=[280000, 360000], GAMMA=0.1, BATCH_SIZE=32, LR=1e-3
)
