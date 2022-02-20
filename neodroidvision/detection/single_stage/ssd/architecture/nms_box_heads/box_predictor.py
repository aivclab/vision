from abc import abstractmethod
from typing import Tuple

import torch
from torch import nn

__all__ = ["BoxPredictor"]


class BoxPredictor(nn.Module):
    """ """

    def __init__(self, boxes_per_location, out_channels, num_categories):
        super().__init__()

        self.num_categories = num_categories
        self.out_channels = out_channels

        self.cls_headers = nn.ModuleList()
        self.reg_headers = nn.ModuleList()

        for (level_i, (num_boxes, num_channels)) in enumerate(
            zip(boxes_per_location, self.out_channels)
        ):
            self.cls_headers.append(
                self.category_block(level_i, num_channels, num_boxes)
            )
            self.reg_headers.append(
                self.location_block(level_i, num_channels, num_boxes)
            )

        self.reset_parameters()

    @abstractmethod
    def category_block(self, level: int, out_channels: int, boxes_per_location: int):
        """

        :param level:
        :type level:
        :param out_channels:
        :type out_channels:
        :param boxes_per_location:
        :type boxes_per_location:"""
        raise NotImplementedError

    @abstractmethod
    def location_block(self, level: int, out_channels: int, boxes_per_location: int):
        """

        :param level:
        :type level:
        :param out_channels:
        :type out_channels:
        :param boxes_per_location:
        :type boxes_per_location:"""
        raise NotImplementedError

    def reset_parameters(self) -> None:
        """ """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param features:
        :type features:
        :return:
        :rtype:"""
        cls_logits = []
        bbox_pred = []
        for feature, cat_fun, loc_fun in zip(
            features, self.cls_headers, self.reg_headers
        ):
            cls_logits.append(cat_fun(feature).permute(0, 2, 3, 1).contiguous())
            bbox_pred.append(loc_fun(feature).permute(0, 2, 3, 1).contiguous())

        batch_size = features[0].shape[0]

        return (
            torch.cat([c.reshape(c.shape[0], -1) for c in cls_logits], dim=1).reshape(
                batch_size, -1, self.num_categories
            ),
            torch.cat([l.reshape(l.shape[0], -1) for l in bbox_pred], dim=1).reshape(
                batch_size, -1, 4
            ),
        )
