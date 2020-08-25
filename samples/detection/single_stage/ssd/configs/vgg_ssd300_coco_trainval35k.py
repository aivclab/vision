from neodroidvision.data.datasets.supervised.detection.coco import COCODataset
from neodroidvision.detection.single_stage.ssd.config.ssd_config import base_cfg

base_cfg.data_dir /= "COCO"

base_cfg.model.box_head.priors.update(
    feature_maps=(38, 19, 10, 5, 3, 1),
    strides=(8, 16, 32, 64, 100, 300),
    min_sizes=(21, 45, 99, 153, 207, 261),
    max_sizes=(45, 99, 153, 207, 261, 315),
    aspect_ratios=((2,), (2, 3), (2, 3), (2, 3), (2,), (2,)),
    boxes_per_location=(4, 6, 6, 6, 4, 4),
)

base_cfg.input.update(image_size=300)

base_cfg.datasets.update(
    train=("coco_2014_train", "coco_2014_valminusminival"), test=("coco_2014_minival",)
)
base_cfg.dataset_type = COCODataset
base_cfg.solver.update(
    max_iter=400000, lr_steps=(280000, 360000), gamma=0.1, batch_size=32, lr=1e-3
)
base_cfg.model.box_head.update(num_categories=len(base_cfg.dataset_type.categories))
