import torch
from torch import nn

__all__ = ["SingleShotDetection"]
__doc__ = r""""""

from warg import NOD


class SingleShotDetection(nn.Module):
    """ """

    def __init__(self, cfg: NOD):
        super().__init__()
        self.backbone = cfg.model.backbone.name(
            cfg.input.image_size, cfg.model.backbone.pretrained
        )

        self.predictor = cfg.model.backbone.predictor_type(
            cfg.model.box_head.priors.boxes_per_location,
            cfg.model.backbone.out_channels,
            cfg.model.box_head.num_categories,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """

        Args:
          images:

        Returns:

        """
        return self.backbone(images)
