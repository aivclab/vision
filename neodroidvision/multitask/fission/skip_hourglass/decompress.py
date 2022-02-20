import torch
from torch import nn

from neodroidvision.multitask.fission.skip_hourglass.modes import MergeMode, UpscaleMode

__all__ = ["Decompress"]


class Decompress(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution."""

    @staticmethod
    def decompress(
        in_channels: int,
        out_channels: int,
        *,
        mode: UpscaleMode = UpscaleMode.FractionalTranspose,
        factor: int = 2,
    ) -> nn.Module:
        """

        :param in_channels:
        :type in_channels:
        :param out_channels:
        :type out_channels:
        :param mode:
        :type mode:
        :param factor:
        :type factor:
        :return:
        :rtype:"""
        if mode == UpscaleMode.FractionalTranspose:
            return nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=factor
            )
        else:
            # out_channels is always going to be the same as in_channels
            return nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=factor, align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=1, stride=1),
            )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        merge_mode: MergeMode = MergeMode.Concat,
        upscale_mode: UpscaleMode = UpscaleMode.FractionalTranspose,
        activation=torch.relu,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = upscale_mode
        self.activation = activation

        self.upconv = self.decompress(
            self.in_channels, self.out_channels, mode=self.up_mode
        )

        if self.merge_mode == MergeMode.Concat:
            self.conv1 = nn.Conv2d(
                2 * self.out_channels,
                self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=1,
            )
        else:  # num of input channels to conv2 is same
            self.conv1 = nn.Conv2d(
                self.out_channels,
                self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=1,
            )

        self.conv2 = nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=1,
        )

    def forward(self, from_down: torch.Tensor, from_up: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Arguments:
        from_down: tensor from the encoder pathway
        from_up: upconv'd tensor from the decoder pathway"""
        from_up = self.upconv(from_up)

        if self.merge_mode == MergeMode.Concat:
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down

        return self.activation(self.conv2(self.activation(self.conv1(x))))
