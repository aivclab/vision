from typing import List

import torch
from torch import nn

__all__ = ["SingleShotDectection"]

from neodroidvision.detection.single_stage.ssd.architecture.box_heads import (
    SSDBoxHead,
    SSDOut,
)

from warg import NOD


class SingleShotDectection(nn.Module):
    def __init__(self, cfg: NOD):
        super().__init__()
        self.backbone = cfg.model.backbone.name(
            cfg.input.image_size, cfg.model.backbone.pretrained
        )

        predictor = cfg.model.backbone.predictor_type(
            cfg.model.box_head.priors.boxes_per_location,
            cfg.model.backbone.out_channels,
            cfg.model.box_head.num_categories,
        )

        self.box_head = SSDBoxHead(
            image_size=cfg.input.image_size, predictor=predictor, **cfg.model.box_head
        )

        self.priors_cfg = cfg.model.box_head.priors

    def forward(self, images: torch.Tensor) -> List[SSDOut]:
        return self.box_head(self.backbone(images))

    def post_init(self) -> None:
        self.box_head.post_init(**self.priors_cfg)
