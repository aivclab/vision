from neodroidvision.detection.single_stage.ssd.config.base_config import base_cfg

base_cfg.MODEL.update(NUM_CLASSES=21)
base_cfg.MODEL.BACKBONE.update(OUT_CHANNELS=(512, 1024, 512, 256, 256, 256, 256))
base_cfg.MODEL.PRIORS.update(
    FEATURE_MAPS=[64, 32, 16, 8, 4, 2, 1],
    STRIDES=[8, 16, 32, 64, 128, 256, 512],
    MIN_SIZES=[35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8],
    MAX_SIZES=[76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.65],
    ASPECT_RATIOS=[[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    BOXES_PER_LOCATION=[4, 6, 6, 6, 6, 4, 4],
)
base_cfg.INPUT.update(IMAGE_SIZE=512)
base_cfg.DATASETS.update(
    TRAIN=("voc_2007_trainval", "voc_2012_trainval"), TEST=("voc_2007_test",)
)
base_cfg.dataset_type = "voc"
base_cfg.SOLVER.update(
    MAX_ITER=120000, LR_STEPS=[80000, 100000], GAMMA=0.1, BATCH_SIZE=24, LR=1e-3
)
