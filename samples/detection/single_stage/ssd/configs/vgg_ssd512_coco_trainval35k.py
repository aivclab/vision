from neodroidvision.detection.single_stage.ssd.config.base_config import base_cfg

base_cfg.MODEL.update(NUM_CLASSES=81)
base_cfg.MODEL.BACKBONE.update(OUT_CHANNELS=(512, 1024, 512, 256, 256, 256, 256))
base_cfg.MODEL.PRIORS.update(
    FEATURE_MAPS=[64, 32, 16, 8, 4, 2, 1],
    STRIDES=[8, 16, 32, 64, 128, 256, 512],
    MIN_SIZES=[20.48, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8],
    MAX_SIZES=[51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],
    ASPECT_RATIOS=[[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    BOXES_PER_LOCATION=[4, 6, 6, 6, 6, 4, 4],
)
base_cfg.INPUT.update(IMAGE_SIZE=512)
base_cfg.DATASETS.update(
    TRAIN=("coco_2014_train", "coco_2014_valminusminival"), TEST=("coco_2014_minival",)
)
base_cfg.dataset_type = "coco"
base_cfg.SOLVER.update(
    MAX_ITER=520000, LR_STEPS=[360000, 480000], GAMMA=0.1, BATCH_SIZE=24, LR=1e-3
)
