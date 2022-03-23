from enum import Enum
from typing import List, Union

from torch import Tensor, nn
from torch.nn.functional import relu

from neodroidvision.detection.single_stage.ssd.architecture.backbones.ssd_backbone import (
    SSDBackbone,
)
from neodroidvision.utilities.torch_utilities import L2Norm


__all__ = ["VGG"]


class VGG(SSDBackbone):
    """ """

    class VggSize(Enum):
        s300 = "300"
        s512 = "512"

    vgg_base = {
        VggSize.s300: [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            "C",
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
        ],
        VggSize.s512: [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            "C",
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
        ],
    }
    extras_base = {
        VggSize.s300: [256, "S", 512, 128, "S", 256, 128, 256, 128, 256],
        VggSize.s512: [256, "S", 512, 128, "S", 256, 128, "S", 256, 128, "S", 256],
    }

    @staticmethod
    def add_vgg(cfg, batch_norm: bool = False, *, in_channels: int = 3):
        """

        Args:
          cfg:
          batch_norm:
          in_channels:

        Returns:

        """
        layers = []

        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == "C":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        layers += [
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True),
        ]
        return layers

    @staticmethod
    def add_extras(cfg, start_in_c_num: Union[str, int], vgg_size: VggSize = 300):
        """
        Extra layers added to VGG for feature scaling

        :param cfg:
        :type cfg:
        :param start_in_c_num:
        :type start_in_c_num:
        :param vgg_size:
        :type vgg_size:
        :return:
        :rtype:"""

        layers = []
        in_channels = start_in_c_num
        flag = False
        for k, v in enumerate(cfg):
            if in_channels != "S":
                if v == "S":
                    layers += [
                        nn.Conv2d(
                            in_channels,
                            cfg[k + 1],
                            kernel_size=(1, 3)[flag],
                            stride=2,
                            padding=1,
                        )
                    ]
                else:
                    layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
                flag = not flag
            in_channels = v

        if vgg_size == VGG.VggSize.s512:
            layers.append(nn.Conv2d(in_channels, 128, kernel_size=1, stride=1))
            layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1))

        return layers

    def __init__(self, vgg_size: VggSize):
        super().__init__(vgg_size)
        if not isinstance(vgg_size, VGG.VggSize):
            vgg_size = VGG.VggSize(str(vgg_size))
        assert isinstance(vgg_size, VGG.VggSize)
        vgg_config = self.vgg_base[vgg_size]
        extras_config = self.extras_base[vgg_size]

        self.vgg = nn.ModuleList(self.add_vgg(vgg_config))
        self.extras = nn.ModuleList(
            self.add_extras(extras_config, start_in_c_num=1024, vgg_size=vgg_size)
        )
        self.l2_norm = L2Norm(512, scale=20)
        self.reset_parameters()

    def init_from_pretrain(self, state_dict):
        """

        Args:
          state_dict:
        """
        self.vgg.load_state_dict(state_dict)

    def forward(self, x: Tensor) -> List[Tensor]:
        """

        Args:
          x:

        Returns:

        """
        features = []

        for i in range(23):
            x = self.vgg[i](x)
        features.append(self.l2_norm(x))  # Conv4_3 L2 normalization

        for i in range(23, len(self.vgg)):  # apply vgg up to fc7
            x = self.vgg[i](x)
        features.append(x)

        for k, v in enumerate(self.extras):
            x = relu(v(x), inplace=True)
            if k % 2 == 1:
                features.append(x)

        return features
