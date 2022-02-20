from typing import Dict, Iterable, Sequence, Tuple, Union

import numpy
import torch
from draugr.torch_utilities import to_tensor
from torch import nn
from torch.nn import init

from neodroidvision.multitask.fission.skip_hourglass.factory import (
    fcn_decoder,
    fcn_encoder,
)
from neodroidvision.multitask.fission.skip_hourglass.modes import MergeMode, UpscaleMode

__all__ = ["SkipHourglassFission"]


class SkipHourglassFission(nn.Module):
    """
    Multi Headed Skip Fully Convolutional Network

    Based on https://arxiv.org/abs/1505.04597
    Contextual spatial information (from the decoding, expansive pathway) about an input tensor is merged
    with information representing the localization of details (from the encoding, compressive pathway).

    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
    pathway (specified by upmode='upsample'), then an
    additional 1x1 2d convolution occurs after upsampling
    to reduce channel dimensionality by a factor of 2.
    This channel halving happens with the convolution in
    the tranpose convolution (specified by upmode='fractional')"""

    def parse_arguments(self, up_mode: UpscaleMode, merge_mode: MergeMode):
        """

        Args:
          up_mode:
          merge_mode:
        """
        if up_mode in UpscaleMode:
            self.up_mode = up_mode
        else:
            raise ValueError(
                f'"{up_mode}" is not a valid mode for upsampling.'
                f' Only "fractional" and "upsample" are allowed.'
            )

        if merge_mode in MergeMode:
            self.merge_mode = merge_mode
        else:
            raise ValueError(
                f'"{up_mode}" is not a valid mode for merging up and down paths.'
                f' Only "concat" and "add" are allowed.'
            )

        if self.up_mode == UpscaleMode.Upsample and self.merge_mode == MergeMode.Add:
            # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
            raise ValueError(
                'up_mode "upsample" is incompatible with merge_mode "add"'
                "can not use nearest neighbour to reduce depth channels (by half)."
            )

    def __init__(
        self,
        *,
        input_channels: int,
        output_heads: Union[Dict, Iterable],
        encoding_depth: int = 5,
        start_channels: int = 32,
        up_mode: UpscaleMode = UpscaleMode.FractionalTranspose,
        merge_mode: MergeMode = MergeMode.Add,
    ):
        """
        :type input_channels: int
        :type output_heads: Union[Dict, Iterable]
        :type encoding_depth: int
        :type start_channels: int
        :type up_mode: str
        :type merge_mode: str

        :param input_channels: number of channels in the input tensor. E.g. 3 for RGB images.
        :param output_heads:
        :param encoding_depth: number of MaxPools in the U-Net.
        :param start_channels: number of convolutional filters for the first convolution.
        :param up_mode: string, type of upconvolution. Choices: 'fractional' for transpose convolution or
        'upsample' for nearest neighbour upsampling.
        :param merge_mode:"""
        super().__init__()

        self.parse_arguments(up_mode, merge_mode)

        self.start_channels = start_channels
        self.network_depth = encoding_depth

        down_convolutions, encoding_channels = fcn_encoder(
            input_channels, self.network_depth, self.start_channels
        )

        self.down_convolutions = nn.ModuleList(down_convolutions)

        self.default_prefix = "fork"
        self._dict_output = False
        if isinstance(output_heads, Dict):
            self.forks = output_heads
            self._dict_output = True
        else:
            self.forks = {
                f"{self.default_prefix}{i}": s for i, s in enumerate(output_heads)
            }

        for iden, channel_size in self.forks.items():
            up_convolutions_ae, ae_prev_layer_channels = fcn_decoder(
                encoding_channels, self.network_depth, self.up_mode, self.merge_mode
            )
            up_convolutions_ae.append(
                nn.Conv2d(
                    ae_prev_layer_channels,
                    channel_size,
                    kernel_size=1,
                    groups=1,
                    stride=1,
                )
            )
            setattr(self, iden, nn.ModuleList(up_convolutions_ae))

        self.reset_params()

    @staticmethod
    def weight_init(m: nn.Module) -> None:
        """

        Args:
          m:
        """
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self) -> None:
        """ """
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(
        self, x_enc: torch.Tensor
    ) -> Union[Tuple[torch.Tensor], Dict[str, torch.Tensor]]:
        """

        Args:
          x_enc:

        Returns:

        """
        encoder_skips = []

        for i, module in enumerate(
            self.down_convolutions
        ):  # encoder pathway, keep outputs for merging
            x_enc, before_pool = module(x_enc)
            encoder_skips.append(before_pool)

        out = {}
        for key in self.forks.keys():
            x_prev = x_enc
            fork_i = getattr(self, key)
            for j, module in enumerate(fork_i[:-1]):
                before_pool_skip = encoder_skips[-(j + 2)]
                x_prev = module(before_pool_skip, x_prev)
            out[key] = fork_i[-1](x_prev)

        if self._dict_output:
            return out
        return (*out.values(),)

    def trim(self, idx: Sequence[Union[str, int]]) -> None:
        """

        Args:
          idx:
        """
        if not isinstance(idx, Sequence):
            idx = list(idx)
        for a in idx:
            if not isinstance(a, str):
                a = f"{self.default_prefix}{a}"
            delattr(self, a)


if __name__ == "__main__":
    from matplotlib import pyplot

    channels = 3
    model = SkipHourglassFission(
        input_channels=channels,
        output_heads=(channels, 1),
        encoding_depth=2,
        merge_mode=MergeMode.Concat,
    )
    x = to_tensor(numpy.random.random((1, channels, 320, 320)), device="cpu").float()
    out, out2, *_ = model(x)
    loss = torch.sum(out)
    loss.backward()

    im = out.detach()
    print(im.shape)
    pyplot.imshow((torch.tanh(im[0].transpose(2, 0)) + 1) * 0.5)
    pyplot.show()

    im2 = out2.detach()
    print(im2.shape)
    pyplot.imshow((torch.tanh(im2[0][0, :, :]) + 1) * 0.5)
    pyplot.show()
