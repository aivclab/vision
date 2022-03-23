from typing import List

from torch import Tensor, nn

from neodroidvision.detection.single_stage.ssd.architecture.backbones.ssd_backbone import (
    SSDBackbone,
)

__all__ = ["MobileNetV2"]


class MobileNetV2(SSDBackbone):
    """ """

    class ConvBatchNormReLU(nn.Sequential):
        """ """

        def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
            padding = (kernel_size - 1) // 2
            super().__init__(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size,
                    stride,
                    padding,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(out_planes),
                nn.ReLU6(inplace=True),
            )

    class InvertedResidual(nn.Module):
        """ """

        def __init__(self, inp, oup, stride, expand_ratio):
            super().__init__()
            self.stride = stride
            assert stride in [1, 2]

            hidden_dim = int(round(inp * expand_ratio))
            self.use_res_connect = self.stride == 1 and inp == oup

            layers = []
            if expand_ratio != 1:
                # pw
                layers.append(
                    MobileNetV2.ConvBatchNormReLU(inp, hidden_dim, kernel_size=1)
                )
            layers.extend(
                [
                    # dw
                    MobileNetV2.ConvBatchNormReLU(
                        hidden_dim, hidden_dim, stride=stride, groups=hidden_dim
                    ),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                ]
            )
            self.conv = nn.Sequential(*layers)

        def forward(self, x: Tensor) -> Tensor:
            """

            Args:
              x:

            Returns:

            """
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.conv(x)

    def __init__(
        self,
        size: int,
        width_mult: float = 1.0,
        inverted_residual_setting=None,
        input_channel: int = 32,
        last_channel: int = 1280,
    ):
        super().__init__(size)
        block = self.InvertedResidual

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if (
            len(inverted_residual_setting) == 0
            or len(inverted_residual_setting[0]) != 4
        ):
            raise ValueError(
                "inverted_residual_setting should be non-empty "
                f"or a 4-element list, got {inverted_residual_setting}"
            )

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel_num = int(last_channel * max(1.0, width_mult))
        feature_extractor = [MobileNetV2.ConvBatchNormReLU(3, input_channel, stride=2)]

        for (
            t,
            c,
            n,
            s,
        ) in inverted_residual_setting:  # building inverted residual blocks
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                feature_extractor.append(
                    block(input_channel, output_channel, stride, expand_ratio=t)
                )
                input_channel = output_channel
        # building last several layers
        feature_extractor.append(
            MobileNetV2.ConvBatchNormReLU(
                input_channel, self.last_channel_num, kernel_size=1
            )
        )
        # make it nn.Sequential
        self.features = nn.Sequential(*feature_extractor)
        self.extras = nn.ModuleList(
            [
                MobileNetV2.InvertedResidual(1280, 512, 2, 0.2),
                MobileNetV2.InvertedResidual(512, 256, 2, 0.25),
                MobileNetV2.InvertedResidual(256, 256, 2, 0.5),
                MobileNetV2.InvertedResidual(256, 64, 2, 0.25),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        """
        weight initialization

        :return:
        :rtype:"""

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> List[Tensor]:
        """

        Args:
          x:

        Returns:

        """
        features = []
        for i in range(14):
            x = self.features[i](x)
        features.append(x)

        for i in range(14, len(self.features)):
            x = self.features[i](x)
        features.append(x)

        for i in range(len(self.extras)):
            x = self.extras[i](x)
            features.append(x)

        return features
