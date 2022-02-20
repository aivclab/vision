from torch import nn

__all__ = ["SeparableConv2d"]


class SeparableConv2d(nn.Module):
    """"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        onnx_compatible: bool = False,
    ):
        """


        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param onnx_compatible:"""
        super().__init__()
        ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                groups=in_channels,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(in_channels),
            ReLU(),
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1
            ),
        )

    def forward(self, x):
        """

        :param x:
        :return:
        """
        return self.conv(x)
