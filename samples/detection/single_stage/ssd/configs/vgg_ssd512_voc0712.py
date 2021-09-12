from data.detection.voc import VOCDataset

from neodroidvision.detection.single_stage.ssd.config.ssd_base_config import base_cfg

base_cfg.data_dir = base_cfg.data_dir / "PASCAL" / "Train"

base_cfg.model.backbone.update(out_channels=(512, 1024, 512, 256, 256, 256, 256))
base_cfg.model.box_head.priors.update(
    feature_maps=(64, 32, 16, 8, 4, 2, 1),
    strides=(8, 16, 32, 64, 128, 256, 512),
    min_sizes=(35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8),
    max_sizes=(76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.65),
    aspect_ratios=((2,), (2, 3), (2, 3), (2, 3), (2, 3), (2,), (2,)),
    boxes_per_location=(4, 6, 6, 6, 6, 4, 4),
)
base_cfg.input.update(image_size=512)
base_cfg.datasets.update(
    train=("voc_2007_trainval", "voc_2012_trainval"), test=("voc_2007_test",)
)
base_cfg.dataset_type = VOCDataset
base_cfg.solver.update(
    max_iter=120000, lr_steps=(80000, 100000), gamma=0.1, batch_size=24, lr=1e-3
)
base_cfg.model.box_head.update(num_categories=len(base_cfg.dataset_type.categories))
