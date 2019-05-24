import numpy as np
import torch
import torch.nn as nn
from torch.nn import init

from vision.segmentation.architectures.fcn import conv1x1, fcn_decoder, fcn_encoder


class MultiHeadedSkipFCN(nn.Module):
  """
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
      the tranpose convolution (specified by upmode='transpose')
  """

  def parse_arguments(self, up_mode, merge_mode):
    if up_mode in ('transpose', 'upsample'):
      self.up_mode = up_mode
    else:
      raise ValueError(
          f'"{up_mode}" is not a valid mode for upsampling. Only "transpose" and "upsample" are allowed.')

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
               segmentation_channels,
               *,
               input_channels=3,
               depth=5,
               start_channels=32,
               up_mode='transpose',
               merge_mode='add'):
    '''
    Arguments:
        input_channels: int, number of channels in the input tensor.
            Default is 3 for RGB images.
        depth: int, number of MaxPools in the U-Net.
        start_channels: int, number of convolutional filters for the            first conv.
        up_mode: string, type of upconvolution.
          Choices: 'transpose' for transpose convolution or
          'upsample' for nearest neighbour upsampling.
        merge_mode:'concat'
    '''
    super().__init__()

    self.parse_arguments(up_mode, merge_mode)

    self.segmentation_channels = segmentation_channels
    self.depth_output_channels = 1
    self.normals_output_channels = 3
    self.input_channels = input_channels
    self.start_channels = start_channels
    self.network_depth = depth

    down_convolutions, encoding_channels = fcn_encoder(self.input_channels,
                                                       self.network_depth,
                                                       self.start_channels)

    up_convolutions_ae, ae_prev_layer_channels = fcn_decoder(encoding_channels,
                                                             self.network_depth,
                                                             self.up_mode,
                                                             self.merge_mode)
    self.ae_head = conv1x1(ae_prev_layer_channels, self.input_channels)

    up_convolutions_seg, seg_prev_layer_channels = fcn_decoder(encoding_channels,
                                                               self.network_depth,
                                                               self.up_mode,
                                                               self.merge_mode)
    self.seg_head = conv1x1(seg_prev_layer_channels, self.segmentation_channels)

    up_convolutions_depth, depth_prev_layer_channels = fcn_decoder(encoding_channels,
                                                                   self.network_depth,
                                                                   self.up_mode,
                                                                   self.merge_mode)
    self.depth_head = conv1x1(depth_prev_layer_channels, self.depth_output_channels)

    up_convolutions_normals, normals_prev_layer_channels = fcn_decoder(encoding_channels,
                                                                       self.network_depth,
                                                                       self.up_mode,
                                                                       self.merge_mode)
    self.normals_head = conv1x1(normals_prev_layer_channels, self.normals_output_channels)

    self.down_convolutions = nn.ModuleList(down_convolutions)
    self.up_convolutions_ae = nn.ModuleList(up_convolutions_ae)
    self.up_convolutions_seg = nn.ModuleList(up_convolutions_seg)
    self.up_convolutions_depth = nn.ModuleList(up_convolutions_depth)
    self.up_convolutions_normals = nn.ModuleList(up_convolutions_normals)

    self.reset_params()

  @staticmethod
  def weight_init(m):
    if isinstance(m, nn.Conv2d):
      init.xavier_normal_(m.weight)
      init.constant_(m.bias, 0)

  def reset_params(self):
    for i, m in enumerate(self.modules()):
      self.weight_init(m)

  def forward(self, x):
    encoder_skips = []

    for i, module in enumerate(self.down_convolutions):  # encoder pathway, save outputs for merging
      x, before_pool = module(x)
      encoder_skips.append(before_pool)

    x_seg = x
    for i, module in enumerate(self.up_convolutions_seg):
      before_pool = encoder_skips[-(i + 2)]
      x_seg = module(before_pool, x_seg)

    x_ae = x
    for i, module in enumerate(self.up_convolutions_ae):
      before_pool = encoder_skips[-(i + 2)]
      x_ae = module(before_pool, x_ae)

    x_depth = x
    for i, module in enumerate(self.up_convolutions_depth):
      before_pool = encoder_skips[-(i + 2)]
      x_depth = module(before_pool, x_depth)

    x_normal = x
    for i, module in enumerate(self.up_convolutions_normals):
      before_pool = encoder_skips[-(i + 2)]
      x_normal = module(before_pool, x_normal)

    (seg, ae, depth, normals) = (self.seg_head(x_seg),
                                 self.ae_head(x_ae),
                                 self.depth_head(x_depth),
                                 self.normals_head(x_normal))

    return seg, ae, depth, normals


if __name__ == "__main__":
  model = MultiHeadedSkipFCN(3, depth=2, merge_mode='concat')
  x = torch.FloatTensor(np.random.random((1, 3, 320, 320)))
  out, _, _, _ = model(x)
  loss = torch.sum(out)
  loss.backward()
  import matplotlib.pyplot as plt

  im = out.detach()
  print(im.shape)
  plt.imshow(im[0].transpose(2, 0))
  plt.show()
