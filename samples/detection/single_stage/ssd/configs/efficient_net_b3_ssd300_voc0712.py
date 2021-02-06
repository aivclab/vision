#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/03/2020
           """

from data.detection.voc import VOCDataset
from neodroidvision.detection import efficient_net_b3_factory
from neodroidvision.detection.single_stage.ssd.config.ssd_base_config import base_cfg

base_cfg.data_dir = base_cfg.data_dir / "PASCAL" / "Train"

base_cfg.model.backbone.update(
    name=efficient_net_b3_factory, out_channels=(48, 136, 384, 256, 256, 256)
)
base_cfg.input.update(image_size=300)
base_cfg.datasets.update(
    train=("voc_2007_trainval", "voc_2012_trainval"), test=("voc_2007_test",)
)
base_cfg.dataset_type = VOCDataset
base_cfg.solver.update(
    max_iter=160000, lr_steps=[105000, 135000], gamma=0.1, batch_size=24, lr=1e-3
)
base_cfg.model.box_head.update(num_categories=len(base_cfg.dataset_type.categories))
