from typing import Sequence

import torch
from torch import nn

__all__ = ["make_san"]

from neodroidvision.mixed.architectures.self_attention_network.enums import (
    PadModeEnum,
    SelfAttentionTypeEnum,
)

from neodroidvision.mixed.architectures.self_attention_network.self_attention_modules.modules import (
    Aggregation,
    Subtraction,
    Subtraction2,
)


class SelfAttentionModule(nn.Module):
    """ """

    def __init__(
        self,
        self_attention_type: SelfAttentionTypeEnum,
        in_planes: int,
        rel_planes: int,
        out_planes: int,
        share_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
    ):
        """

        :param self_attention_type:
        :type self_attention_type:
        :param in_planes:
        :type in_planes:
        :param rel_planes:
        :type rel_planes:
        :param out_planes:
        :type out_planes:
        :param share_planes:
        :type share_planes:
        :param kernel_size:
        :type kernel_size:
        :param stride:
        :type stride:
        :param dilation:
        :type dilation:"""
        super().__init__()
        self.self_attention_type, self.kernel_size, self.stride = (
            self_attention_type,
            kernel_size,
            stride,
        )
        self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)

        pad_mode = PadModeEnum.ref_pad
        if self_attention_type == SelfAttentionTypeEnum.pairwise:
            self.conv_w = nn.Sequential(
                nn.BatchNorm2d(rel_planes + 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(rel_planes + 2, rel_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(rel_planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(rel_planes, out_planes // share_planes, kernel_size=1),
            )
            self.conv_p = nn.Conv2d(2, 2, kernel_size=1)
            self.subtraction = Subtraction(
                kernel_size,
                stride,
                (dilation * (kernel_size - 1) + 1) // 2,
                dilation,
                pad_mode=pad_mode,
            )
            self.subtraction2 = Subtraction2(
                kernel_size,
                stride,
                (dilation * (kernel_size - 1) + 1) // 2,
                dilation,
                pad_mode=pad_mode,
            )
            self.softmax = nn.Softmax(
                dim=-2
            )  # TODO: USE log softmax? Check dim maybe it should be 1
        elif self_attention_type == SelfAttentionTypeEnum.patchwise:
            self.conv_w = nn.Sequential(
                nn.BatchNorm2d(rel_planes * (kernel_size**2 + 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    rel_planes * (kernel_size**2 + 1),
                    out_planes // share_planes,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_planes // share_planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_planes // share_planes,
                    kernel_size**2 * out_planes // share_planes,
                    kernel_size=1,
                ),
            )
            self.unfold_i = nn.Unfold(
                kernel_size=1, dilation=dilation, padding=0, stride=stride
            )
            self.unfold_j = nn.Unfold(
                kernel_size=kernel_size, dilation=dilation, padding=0, stride=stride
            )
            self.pad = nn.ReflectionPad2d(kernel_size // 2)
        else:
            raise NotImplemented(
                f"self_attention_type {self_attention_type} not supported"
            )

        self.aggregation = Aggregation(
            kernel_size,
            stride,
            (dilation * (kernel_size - 1) + 1) // 2,
            dilation,
            pad_mode=pad_mode,
        )

    def encode_position(
        self, height: int, width: int, is_cuda: bool = True
    ) -> torch.Tensor:
        """

        :param height:
        :type height:
        :param width:
        :type width:
        :param is_cuda:
        :type is_cuda:
        :return:
        :rtype:"""
        if is_cuda:
            loc_w = (
                torch.linspace(-1.0, 1.0, width).cuda().unsqueeze(0).repeat(height, 1)
            )
            loc_h = (
                torch.linspace(-1.0, 1.0, height).cuda().unsqueeze(1).repeat(1, width)
            )
        else:
            loc_w = torch.linspace(-1.0, 1.0, width).unsqueeze(0).repeat(height, 1)
            loc_h = torch.linspace(-1.0, 1.0, height).unsqueeze(1).repeat(1, width)
        return torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :type x:
        :return:
        :rtype:"""
        x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x)
        if self.self_attention_type == SelfAttentionTypeEnum.pairwise:
            position = self.conv_p(self.encode_position(*x.shape[2:], x.is_cuda))
            w = self.softmax(
                self.conv_w(
                    torch.cat(
                        [
                            self.subtraction2(x1, x2),
                            self.subtraction(position).repeat(x.shape[0], 1, 1, 1),
                        ],
                        1,
                    )
                )
            )
        elif self.self_attention_type == SelfAttentionTypeEnum.patchwise:
            if self.stride != 1:
                x1 = self.unfold_i(x1)
            x1 = x1.view(x.shape[0], -1, 1, torch.prod(x.shape[2:]).item())
            x2 = self.unfold_j(self.pad(x2)).view(x.shape[0], -1, 1, x1.shape[-1])
            w = self.conv_w(torch.cat([x1, x2], 1)).view(
                x.shape[0], -1, self.kernel_size**2, x1.shape[-1]
            )
        else:
            raise NotImplemented()
        return self.aggregation(x3, w)


class SelfAttentionBottleneck(nn.Module):
    """ """

    def __init__(
        self,
        self_attention_type: SelfAttentionTypeEnum,
        in_planes: int,
        rel_planes: int,
        mid_planes: int,
        out_planes: int,
        share_planes: int = 8,
        kernel_size: int = 7,
        stride: int = 1,
    ):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.sam = SelfAttentionModule(
            self_attention_type,
            in_planes,
            rel_planes,
            mid_planes,
            share_planes,
            kernel_size,
            stride,
        )
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv = nn.Conv2d(mid_planes, out_planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :type x:
        :return:
        :rtype:"""
        identity = x
        out = self.relu(self.bn1(x))
        out = self.relu(self.bn2(self.sam(out)))
        out = self.conv(out)
        out += identity
        return out


class SelfAttentionNetwork(nn.Module):
    """ """

    @staticmethod
    def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Module:
        """

        :param in_planes:
        :type in_planes:
        :param out_planes:
        :type out_planes:
        :param stride:
        :type stride:
        :return:
        :rtype:"""
        return nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride, bias=False
        )

    def __init__(
        self,
        self_attention_type: SelfAttentionTypeEnum,
        block,
        layers,
        kernels,
        num_classes,
    ):
        super().__init__()
        c = 64
        self.conv_in, self.bn_in = self.conv1x1(3, c), nn.BatchNorm2d(c)
        self.conv0, self.bn0 = self.conv1x1(c, c), nn.BatchNorm2d(c)
        self.layer0 = self._make_layer(
            self_attention_type, block, c, layers[0], kernels[0]
        )

        c *= 4
        self.conv1, self.bn1 = self.conv1x1(c // 4, c), nn.BatchNorm2d(c)
        self.layer1 = self._make_layer(
            self_attention_type, block, c, layers[1], kernels[1]
        )

        c *= 2
        self.conv2, self.bn2 = self.conv1x1(c // 2, c), nn.BatchNorm2d(c)
        self.layer2 = self._make_layer(
            self_attention_type, block, c, layers[2], kernels[2]
        )

        c *= 2
        self.conv3, self.bn3 = self.conv1x1(c // 2, c), nn.BatchNorm2d(c)
        self.layer3 = self._make_layer(
            self_attention_type, block, c, layers[3], kernels[3]
        )

        c *= 2
        self.conv4, self.bn4 = self.conv1x1(c // 2, c), nn.BatchNorm2d(c)
        self.layer4 = self._make_layer(
            self_attention_type, block, c, layers[4], kernels[4]
        )

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c, num_classes)

    @staticmethod
    def _make_layer(
        self_attention_type: SelfAttentionTypeEnum,
        block: callable,
        planes: int,
        num_blocks: int,
        kernel_size: int = 7,
        stride: int = 1,
    ) -> nn.Module:
        layers = []
        for _ in range(0, num_blocks):
            layers.append(
                block(
                    self_attention_type,
                    planes,
                    planes // 16,
                    planes // 4,
                    planes,
                    8,
                    kernel_size,
                    stride,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :type x:
        :return:
        :rtype:"""
        x = self.relu(self.bn_in(self.conv_in(x)))
        x = self.relu(self.bn0(self.layer0(self.conv0(self.pool(x)))))
        x = self.relu(self.bn1(self.layer1(self.conv1(self.pool(x)))))
        x = self.relu(self.bn2(self.layer2(self.conv2(self.pool(x)))))
        x = self.relu(self.bn3(self.layer3(self.conv3(self.pool(x)))))
        x = self.relu(self.bn4(self.layer4(self.conv4(self.pool(x)))))

        x = self.avgpool(x)
        return self.fc(x.view(x.size(0), -1))


def make_san(
    *,
    self_attention_type: SelfAttentionTypeEnum = SelfAttentionTypeEnum.pairwise,
    layers: Sequence,
    kernels: Sequence,
    num_classes: int,
) -> nn.Module:
    """

    :param self_attention_type:
    :type self_attention_type:
    :param layers:
    :type layers:
    :param kernels:
    :type kernels:
    :param num_classes:
    :type num_classes:
    :return:
    :rtype:"""
    return SelfAttentionNetwork(
        self_attention_type, SelfAttentionBottleneck, layers, kernels, num_classes
    )


if __name__ == "__main__":
    net = (
        make_san(
            self_attention_type=SelfAttentionTypeEnum.pairwise,
            layers=(3, 4, 6, 8, 3),
            kernels=(3, 7, 7, 7, 7),
            num_classes=3,
        )
        .cuda()
        .eval()
    )
    print(net)
    y = net(torch.randn(2, 3, 111, 111).cuda())
    print(y.size())
