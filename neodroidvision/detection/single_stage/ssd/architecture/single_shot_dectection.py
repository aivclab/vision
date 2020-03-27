from torch import nn

__all__ = ["SingleShotDectection"]

from neodroidvision.detection.single_stage.ssd.architecture.backbones import (
    mobilenet_v2_factory,
)
from neodroidvision.detection.single_stage.ssd.architecture.box_heads import (
    SSDLiteBoxPredictor,
    SSDBoxPredictor,
)
from neodroidvision.detection.single_stage.ssd.architecture.box_heads.box_head import (
    SSDBoxHead,
)
from warg import NOD


class SingleShotDectection(nn.Module):
    def __init__(self, cfg: NOD):
        super().__init__()
        self.backbone = cfg.MODEL.BACKBONE.NAME(
            cfg.INPUT.IMAGE_SIZE, cfg.MODEL.BACKBONE.PRETRAINED
        )

        if cfg.MODEL.BACKBONE.NAME == mobilenet_v2_factory:
            predictor_type = SSDLiteBoxPredictor
        else:
            predictor_type = SSDBoxPredictor

        predictor = predictor_type(
            cfg.MODEL.PRIORS.BOXES_PER_LOCATION,
            cfg.MODEL.BACKBONE.OUT_CHANNELS,
            cfg.MODEL.NUM_CLASSES,
        )

        self.box_head = SSDBoxHead(
            IMAGE_SIZE=cfg.INPUT.IMAGE_SIZE,
            CONFIDENCE_THRESHOLD=cfg.TEST.CONFIDENCE_THRESHOLD,
            NMS_THRESHOLD=cfg.TEST.NMS_THRESHOLD,
            MAX_PER_IMAGE=cfg.TEST.MAX_PER_IMAGE,
            PRIORS=cfg.MODEL.PRIORS,
            CENTER_VARIANCE=cfg.MODEL.CENTER_VARIANCE,
            SIZE_VARIANCE=cfg.MODEL.SIZE_VARIANCE,
            predictor=predictor,
        )

    def forward(self, images):
        return self.box_head(self.backbone(images))

    def post_init(self):
        self.box_head.post_init()
