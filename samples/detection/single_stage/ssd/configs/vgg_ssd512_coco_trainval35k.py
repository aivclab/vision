#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/03/2020
           """

from data.detection.coco import COCODataset

from neodroidvision.detection.single_stage.ssd.config.ssd_base_config import base_cfg

base_cfg.data_dir /= "COCO"

base_cfg.dataset_type = COCODataset

base_cfg.model.backbone.update(out_channels=(512, 1024, 512, 256, 256, 256, 256))
base_cfg.model.box_head.priors.update(
    feature_maps=(64, 32, 16, 8, 4, 2, 1),
    strides=(8, 16, 32, 64, 128, 256, 512),
    min_sizes=(20.48, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8),
    max_sizes=(51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72),
    aspect_ratios=((2,), (2, 3), (2, 3), (2, 3), (2, 3), (2,), (2,)),
    boxes_per_location=(4, 6, 6, 6, 6, 4, 4),
    )
base_cfg.input.update(image_size=512)
base_cfg.datasets.update(
    train=("coco_2014_train", "coco_2014_valminusminival"), test=("coco_2014_minival",)
    )

base_cfg.solver.update(
    max_iter=520000, lr_steps=(360000, 480000), gamma=0.1, batch_size=24, lr=1e-3
    )
base_cfg.model.box_head.update(num_categories=len(base_cfg.dataset_type.categories))
