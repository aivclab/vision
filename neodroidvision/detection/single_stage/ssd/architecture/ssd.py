from torch import nn

__all__ = ["SSD"]


class SSD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = cfg.MODEL.BACKBONE.NAME(cfg, cfg.MODEL.BACKBONE.PRETRAINED)
        self.box_head = cfg.MODEL.BOX_HEAD.HEAD(cfg)

    def forward(self, images, targets=None):
        features = self.backbone(images)
        detections, detector_losses = self.box_head(features, targets)
        if self.training:
            return detector_losses
        return detections
