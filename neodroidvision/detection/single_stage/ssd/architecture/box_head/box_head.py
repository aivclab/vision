import torch.nn.functional as F
from torch import nn

from neodroidvision.detection.single_stage.ssd.architecture.anchors.prior_box import (
    PriorBox,
)
from neodroidvision.detection.single_stage.ssd.ssd_utilities.bounding_box import (
    box_utils,
)
from .inference import PostProcessor
from .loss import MultiBoxLoss


class SSDBoxHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.predictor = cfg.MODEL.BOX_HEAD.PREDICTOR(cfg)
        self.loss_evaluator = MultiBoxLoss(neg_pos_ratio=cfg.MODEL.NEG_POS_RATIO)
        self.post_processor = PostProcessor(cfg)
        self.priors = None

    def forward(self, features, targets=None):
        cls_logits, bbox_pred = self.predictor(features)
        if self.training:
            return self._forward_train(cls_logits, bbox_pred, targets)
        else:
            return self._forward_test(cls_logits, bbox_pred)

    def _forward_train(self, cls_logits, bbox_pred, targets):
        gt_boxes, gt_labels = targets["boxes"], targets["labels"]
        reg_loss, cls_loss = self.loss_evaluator(
            cls_logits, bbox_pred, gt_labels, gt_boxes
        )
        loss_dict = dict(reg_loss=reg_loss, cls_loss=cls_loss)
        detections = (cls_logits, bbox_pred)
        return detections, loss_dict

    def _forward_test(self, cls_logits, bbox_pred):
        if self.priors is None:
            self.priors = PriorBox(self.cfg)().to(bbox_pred.device)
        scores = F.softmax(cls_logits, dim=2)
        boxes = box_utils.convert_locations_to_boxes(
            bbox_pred,
            self.priors,
            self.cfg.MODEL.CENTER_VARIANCE,
            self.cfg.MODEL.SIZE_VARIANCE,
        )
        boxes = box_utils.center_form_to_corner_form(boxes)
        detections = (scores, boxes)
        detections = self.post_processor(detections)
        return detections, {}
