from neodroidvision.data.detection.voc import VOCDataset
from neodroidvision.detection.single_stage.ssd.config.ssd_base_config import base_cfg

base_cfg.data_dir = base_cfg.data_dir / "PASCAL" / "Train"

base_cfg.input.update(image_size=300)
base_cfg.datasets.update(
    train=("voc_2007_trainval", "voc_2012_trainval"), test=("voc_2007_test",)
    )
base_cfg.dataset_type = VOCDataset
base_cfg.solver.update(
    max_iter=120000, lr_steps=(80000, 100000), gamma=0.1, batch_size=32, lr=1e-3
    )
base_cfg.model.box_head.update(num_categories=len(base_cfg.dataset_type.categories))
