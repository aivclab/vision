import torch
import torch.nn as nn


class Decompress(nn.Module):
  """
  A helper Module that performs 2 convolutions and 1 UpConvolution.
  A ReLU activation follows each convolution.
  """

  @staticmethod
  def decompress(in_channels,
                 out_channels,
                 mode='fractional',
                 factor=2):
    if mode == 'fractional':
      return nn.ConvTranspose2d(in_channels,
                                out_channels,
                                kernel_size=2,
                                stride=factor)
    else:
      # out_channels is always going to be the same as in_channels
      return nn.Sequential(nn.Upsample(mode='bilinear',
                                       scale_factor=factor,
                                       align_corners=True),
                           nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=1,
                                     groups=1,
                                     stride=1)
                           )

  def __init__(self, in_channels,
               out_channels,
               merge_mode='concat',
               up_mode='fractional'):
    super().__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.merge_mode = merge_mode
    self.up_mode = up_mode

    self.upconv = self.decompress(self.in_channels, self.out_channels, mode=self.up_mode)

    if self.merge_mode == 'concat':
      self.conv1 = nn.Conv2d(2 * self.out_channels,
                             self.out_channels,
                             kernel_size=3,
                             stride=1,
                             padding=1,
                             bias=True,
                             groups=1)
    else:  # num of input channels to conv2 is same
      self.conv1 = nn.Conv2d(self.out_channels,
                             self.out_channels,
                             kernel_size=3,
                             stride=1,
                             padding=1,
                             bias=True,
                             groups=1)

    self.conv2 = nn.Conv2d(self.out_channels,
                           self.out_channels,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           bias=True,
                           groups=1)

  def forward(self, from_down, from_up):
    """
    Forward pass
    Arguments:
        from_down: tensor from the encoder pathway
        from_up: upconv'd tensor from the decoder pathway
    """
    from_up = self.upconv(from_up)

    if self.merge_mode == 'concat':
      x = torch.cat((from_up, from_down), 1)
    else:
      x = from_up + from_down

    x = torch.relu(self.conv1(x))
    x = torch.relu(self.conv2(x))
    return x
