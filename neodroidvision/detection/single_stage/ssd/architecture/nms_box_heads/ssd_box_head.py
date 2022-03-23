from collections import namedtuple
from typing import Any, Tuple

import torch
from draugr.torch_utilities import to_tensor
from torch import nn
from torch.nn import Parameter, functional

from neodroidvision.detection.single_stage.ssd.architecture.nms_box_heads.box_predictor import (
    BoxPredictor,
)
from neodroidvision.detection.single_stage.ssd.bounding_boxes import conversion
from neodroidvision.detection.single_stage.ssd.bounding_boxes.ssd_priors import (
    build_priors,
)

__all__ = ["SSDNmsBoxHead", "SSDOut"]

from neodroidvision.utilities.torch_utilities.output_activation.ops.non_maximum_suppression import (
    batched_non_maximum_suppression,
)
from warg import drop_unused_kws

SSDOut = namedtuple("SSDOut", ("boxes", "labels", "scores", "img_width", "img_height"))


class SSDNmsBoxHead(nn.Module):
    """ """

    @drop_unused_kws
    def __init__(
        self,
        *,
        image_size: Any,
        predictor: BoxPredictor,
        confidence_threshold: Any,
        nms_threshold: Any,
        max_per_image: Any,
        center_variance: Any,
        size_variance: Any,
        max_candidates: int = 100
    ):
        """

        :param image_size:
        :type image_size:
        :param confidence_threshold:
        :type confidence_threshold:
        :param nms_threshold:
        :type nms_threshold:
        :param max_per_image:
        :type max_per_image:
        :param priors:
        :type priors:
        :param center_variance:
        :type center_variance:
        :param size_variance:
        :type size_variance:
        :param predictor:
        :type predictor:"""
        super().__init__()

        self.predictor = predictor

        self.image_size = image_size
        self.width = image_size
        self.height = image_size
        self.confidence_threshold = confidence_threshold
        self.non_maximum_supression_threshold = nms_threshold
        self.max_per_image = max_per_image
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.max_candidates = max_candidates

    def post_init(self, **priors_cfg) -> None:
        """
        Builds priors"""
        self._priors = Parameter(
            build_priors(image_size=self.image_size, **priors_cfg), requires_grad=False
        )

    @staticmethod
    def keep_above(
        scores: torch.Tensor, *args, threshold: float
    ) -> Tuple[torch.Tensor, ...]:
        """
        Only keep detections above the confidence threshold"""
        indices = torch.nonzero(scores > threshold, as_tuple=False).squeeze(
            1
        )  # remove low scoring boxes
        return (scores[indices], *[a[indices] for a in args])

    @staticmethod
    def sort_keep_top_k(
        scores: torch.Tensor, *args, k: int
    ) -> Tuple[torch.Tensor, ...]:
        """
        Only keep detections above the confidence threshold"""
        indices = torch.argsort(scores, descending=True)
        return (scores[indices][:k], *[a[indices][:k] for a in args])

    def forward(self, features: torch.Tensor) -> SSDOut:
        """

        :param features:
        :type features:
        :return:
        :rtype:"""
        categori_logits, bbox_pred = self.predictor(features)

        results = []
        for (scores, boxes) in zip(
            functional.log_softmax(
                categori_logits, dim=-1
            ),  # TODO:Check dim maybe it should be 1
            conversion.center_to_corner_form(
                conversion.locations_to_boxes(
                    locations=bbox_pred,
                    priors=self._priors,
                    center_variance=self.center_variance,
                    size_variance=self.size_variance,
                )
            ),
        ):
            # (N, #CLS) (N, 4)
            num_boxes, num_categories, *_ = scores.shape
            boxes = boxes.reshape(num_boxes, 1, 4).expand(num_boxes, num_categories, 4)
            labels = (
                torch.arange(num_categories, device=scores.device)
                .reshape(1, num_categories)
                .expand_as(scores)
            )

            # remove predictions with the background label and batch everything,
            # by making every class prediction be a separate instance
            boxes = boxes[:, 1:].reshape(-1, 4)
            labels = labels[:, 1:].reshape(-1)
            scores = scores[:, 1:].reshape(-1)

            """ WILL NOT WORK FOR TRACED MODELS!
scores, boxes, labels = self.keep_above(scores,
boxes,
labels,
threshold=self.confidence_threshold)

"""
            scores, boxes, labels = self.sort_keep_top_k(
                scores, boxes, labels, k=self.max_candidates
            )

            # Resize boxes
            boxes[:, 0::2] *= self.width
            boxes[:, 1::2] *= self.height

            keep = batched_non_maximum_suppression(
                boxes, scores, labels, self.non_maximum_supression_threshold
            )
            keep = keep[: self.max_per_image]  # keep only topk scoring predictions

            results.append(
                SSDOut(
                    boxes=boxes[keep],
                    labels=labels[keep],
                    scores=scores[keep],
                    img_width=to_tensor(self.width),
                    img_height=to_tensor(self.height),
                )
            )

        return SSDOut(*[to_tensor(x) for x in zip(*results)])
