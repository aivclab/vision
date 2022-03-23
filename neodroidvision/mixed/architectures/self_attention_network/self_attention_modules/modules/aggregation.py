import torch
from torch import nn
from torch.nn.modules.utils import _pair

__all__ = ["Aggregation"]

from neodroidvision.mixed.architectures.self_attention_network.self_attention_modules.functional import (
    aggregation,
)
from neodroidvision.mixed.architectures.self_attention_network.enums import PadModeEnum


class Aggregation(nn.Module):
    """ """

    def __init__(
        self,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        pad_mode: PadModeEnum,
    ):
        """

        :param kernel_size:
        :type kernel_size:
        :param stride:
        :type stride:
        :param padding:
        :type padding:
        :param dilation:
        :type dilation:
        :param pad_mode:
        :type pad_mode:"""
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.pad_mode = pad_mode

    def forward(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """

        :param input:
        :type input:
        :param weight:
        :type weight:
        :return:
        :rtype:"""
        return aggregation(
            input,
            weight,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.pad_mode,
        )
