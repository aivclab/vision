import torch
from torch import nn
from torch.nn import Parameter, functional

from neodroidvision.detection.single_stage.ssd.bounding_boxes import conversion
from neodroidvision.detection.single_stage.ssd.bounding_boxes.ssd_priors import (
    init_prior_box,
)

__all__ = ["SSDBoxHead"]

from neodroidvision.utilities.torch_utilities.non_maximum_suppression import (
    batched_non_maximum_suppression,
)


class SSDBoxHead(nn.Module):
    def __init__(
        self,
        IMAGE_SIZE,
        CONFIDENCE_THRESHOLD,
        NMS_THRESHOLD,
        MAX_PER_IMAGE,
        PRIORS,
        CENTER_VARIANCE,
        SIZE_VARIANCE,
        predictor,
    ):
        super().__init__()

        self.predictor = predictor

        self.image_size = IMAGE_SIZE
        self.width = IMAGE_SIZE
        self.height = IMAGE_SIZE
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.nms_threshold = NMS_THRESHOLD
        self.max_per_image = MAX_PER_IMAGE
        self.priors_cfg = PRIORS
        self.center_variance = CENTER_VARIANCE
        self.size_variance = SIZE_VARIANCE

    def post_init(self):
        self._priors = Parameter(
            init_prior_box(self.image_size, self.priors_cfg), requires_grad=False
        )

    def forward(self, features):
        cls_logits, bbox_pred = self.predictor(features)

        scores_batch = functional.softmax(cls_logits, dim=-1)
        boxes_batch = conversion.center_form_to_corner_form(
            conversion.convert_locations_to_boxes(
                bbox_pred, self._priors, self.center_variance, self.size_variance
            )
        )

        results = []
        for scores, boxes in zip(scores_batch, boxes_batch):
            # (N, #CLS) (N, 4)
            num_boxes, num_classes, *_ = scores.shape
            boxes = boxes.view(num_boxes, 1, 4).expand(num_boxes, num_classes, 4)
            labels = (
                torch.arange(num_classes, device=scores.device)
                .view(1, num_classes)
                .expand_as(scores)
            )

            # remove predictions with the background label and batch everything, by making every class prediction
            # be a separate instance
            boxes = boxes[:, 1:].reshape(-1, 4)
            labels = labels[:, 1:].reshape(-1)
            scores = scores[:, 1:].reshape(-1)

            # Only keep detections above the confidence threshold
            indices = torch.nonzero(scores > self.confidence_threshold).squeeze(
                1
            )  # remove low scoring boxes
            boxes, scores, labels = boxes[indices], scores[indices], labels[indices]

            # Resize boxes
            boxes[:, 0::2] *= self.width
            boxes[:, 1::2] *= self.height

            # keep only topk scoring predictions
            keep = batched_non_maximum_suppression(
                boxes, scores, labels, self.nms_threshold
            )[: self.max_per_image]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            results.append(
                dict(
                    boxes=boxes,
                    labels=labels,
                    scores=scores,
                    img_width=self.width,
                    img_height=self.height,
                )
            )
        return results
