from neodroidvision.detection.single_stage.ssd.config.base_config import base_cfg

base_cfg.MODEL.update(NUM_CLASSES=21)
base_cfg.MODEL.BACKBONE.update(
    NAME="efficient_net-b3", OUT_CHANNELS=(48, 136, 384, 256, 256, 256)
)
base_cfg.INPUT.update(IMAGE_SIZE=300)
base_cfg.DATASETS.update(
    TRAIN=("voc_2007_trainval", "voc_2012_trainval"), TEST=("voc_2007_test",)
)
base_cfg.dataset_type = "voc"
base_cfg.SOLVER.update(
    MAX_ITER=160000, LR_STEPS=[105000, 135000], GAMMA=0.1, BATCH_SIZE=24, LR=1e-3
)
