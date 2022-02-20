from typing import List

import torch

__all__ = ["SingleShotDetectionNms"]

from neodroidvision.detection.single_stage.ssd.architecture.single_shot_dectection import (
    SingleShotDetection,
)
from neodroidvision.detection.single_stage.ssd.architecture.nms_box_heads import (
    SSDNmsBoxHead,
    SSDOut,
)

from warg import NOD


class SingleShotDetectionNms(SingleShotDetection):
    """ """

    def __init__(self, cfg: NOD):
        super().__init__(cfg)

        self.box_head = SSDNmsBoxHead(
            image_size=cfg.input.image_size,
            predictor=self.predictor,
            **cfg.model.box_head
        )

        self.priors_cfg = cfg.model.box_head.priors

    def forward(self, images: torch.Tensor) -> List[SSDOut]:
        """

        Args:
          images:

        Returns:

        """
        return self.box_head(super().forward(images))

    def post_init(self) -> None:
        """ """
        self.box_head.post_init(**self.priors_cfg)
