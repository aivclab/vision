from neodroidvision.detection.single_stage.ssd.config.base_config import base_cfg

base_cfg.MODEL.update(NUM_CLASSES=21)
base_cfg.INPUT.update(IMAGE_SIZE=300)
base_cfg.DATASETS.update(
    TRAIN=("voc_2007_trainval", "voc_2012_trainval"), TEST=("voc_2007_test",)
)
base_cfg.dataset_type = "voc"
base_cfg.SOLVER.update(
    MAX_ITER=120000, LR_STEPS=[80000, 100000], GAMMA=0.1, BATCH_SIZE=32, LR=1e-3
)
