from typing import Iterable

import numpy
import torch
import torch.nn as nn
from torch.nn import init

from .decompress import Decompress
from .compress import Compress


def fcn_decoder(in_channels, depth, up_mode, merge_mode):
  up_convolutions_ae = []
  ae_prev_layer_channels = in_channels
  for i in range(depth - 1):
    # create the decoder pathway and add to a list - careful! decoding only requires depth-1 blocks
    ae_new_layer_channels = ae_prev_layer_channels // 2
    up_conv = Decompress(ae_prev_layer_channels,
                         ae_new_layer_channels,
                         up_mode=up_mode,
                         merge_mode=merge_mode)
    ae_prev_layer_channels = ae_new_layer_channels
    up_convolutions_ae.append(up_conv)

  return up_convolutions_ae, ae_prev_layer_channels


def fcn_encoder(in_channels, depth, start_channels):
  down_convolutions = []
  new_layer_channels = start_channels
  prev_layer_channels = in_channels
  for i in range(depth):
    pooling = True if i < depth - 1 else False
    new_layer_channels = new_layer_channels * 2
    down_conv = Compress(prev_layer_channels, new_layer_channels, pooling=pooling)
    prev_layer_channels = new_layer_channels
    down_convolutions.append(down_conv)

  return down_convolutions, prev_layer_channels


class MultiHeadedSkipFCN(nn.Module):
  """
  Multi Headed Skip Fully Convolutional Network

  `UNet` class is based on https://arxiv.org/abs/1505.04597
  Contextual spatial information (from the decoding, expansive pathway) about an input tensor is merged with
  information representing the localization of details (from the encoding, compressive pathway).

  Modifications to the original paper:
  (1) padding is used in 3x3 convolutions to prevent loss of border pixels
  (2) merging outputs does not require cropping due to (1)
  (3) residual connections can be used by specifying UNet(merge_mode='add')
  (4) if non-parametric upsampling is used in the decoder
      pathway (specified by upmode='upsample'), then an
      additional 1x1 2d convolution occurs after upsampling
      to reduce channel dimensionality by a factor of 2.
      This channel halving happens with the convolution in
      the tranpose convolution (specified by upmode='fractional')
  """

  def parse_arguments(self, up_mode: str, merge_mode: str):
    if up_mode in ('fractional', 'upsample'):
      self.up_mode = up_mode
    else:
      raise ValueError(
        f'"{up_mode}" is not a valid mode for upsampling. Only "fractional" and "upsample" are allowed.')

    if merge_mode in ('concat', 'add'):
      self.merge_mode = merge_mode
    else:
      raise ValueError(
        f'"{up_mode}" is not a valid mode for merging up and down paths. Only "concat" and "add" are '
        f'allowed.')

    if self.up_mode == 'upsample' and self.merge_mode == 'add':
      # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
      raise ValueError('up_mode "upsample" is incompatible with merge_mode "add" at the moment because '
                       'it does not make sense to use nearest neighbour to reduce depth channels (by half).')

  def __init__(self,
               input_channels: int,
               output_channels: Iterable[int],
               *,
               encoding_depth: int = 5,
               start_channels: int = 32,
               up_mode: str = 'fractional',
               merge_mode: str = 'add'):
    '''
    Arguments:
        input_channels: int, number of channels in the input tensor.
            Default is 3 for RGB images.
        encoding_depth: int, number of MaxPools in the U-Net.
        start_channels: int, number of convolutional filters for the            first conv.
        up_mode: string, type of upconvolution.
          Choices: 'fractional' for transpose convolution or
          'upsample' for nearest neighbour upsampling.
        merge_mode:'concat'
    '''
    super().__init__()

    self.parse_arguments(up_mode, merge_mode)

    self.start_channels = start_channels
    self.network_depth = encoding_depth
    self._output_channels = output_channels

    down_convolutions, encoding_channels = fcn_encoder(input_channels,
                                                       self.network_depth,
                                                       self.start_channels)

    self.down_convolutions = nn.ModuleList(down_convolutions)

    for i, channel_size in enumerate(output_channels):
      up_convolutions_ae, ae_prev_layer_channels = fcn_decoder(encoding_channels,
                                                               self.network_depth,
                                                               self.up_mode,
                                                               self.merge_mode)
      setattr(self, f'decompress{i}', nn.ModuleList(up_convolutions_ae))

      setattr(self, f'head{i}', nn.Conv2d(ae_prev_layer_channels,
                                          channel_size,
                                          kernel_size=1,
                                          groups=1,
                                          stride=1))

    self.reset_params()

  @staticmethod
  def weight_init(m: nn.Module):
    if isinstance(m, nn.Conv2d):
      init.xavier_normal_(m.weight)
      init.constant_(m.bias, 0)

  def reset_params(self):
    for i, m in enumerate(self.modules()):
      self.weight_init(m)

  def forward(self, x: torch.Tensor):
    encoder_skips = []

    for i, module in enumerate(self.down_convolutions):  # encoder pathway, save outputs for merging
      x, before_pool = module(x)
      encoder_skips.append(before_pool)

    out = []
    for i, _ in enumerate(self._output_channels):
      x_seg = x
      for j, module in enumerate(getattr(self, f'decompress{i}')):
        before_pool = encoder_skips[-(j + 2)]
        x_seg = module(before_pool, x_seg)
      act = getattr(self, f'head{i}')(x_seg)
      out.append(act)

    return (*out,)


if __name__ == "__main__":
  channels = 3
  model = MultiHeadedSkipFCN(input_channels=channels, output_channels=(channels,), encoding_depth=2,
                             merge_mode='concat')
  x = torch.FloatTensor(numpy.random.random((1, channels, 320, 320)))
  out, *_ = model(x)
  loss = torch.sum(out)
  loss.backward()
  from matplotlib import pyplot

  im = out.detach()
  print(im.shape)
  pyplot.imshow(torch.tanh(im[0].transpose(2, 0)) * 0.5 + 1)
  pyplot.show()
