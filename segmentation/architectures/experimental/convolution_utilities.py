import torch.nn as nn


class conv2DBatchNorm(nn.Module):
  def __init__(self,
               in_channels,
               n_filters,
               k_size,
               stride,
               padding,
               bias=True,
               dilation=1,
               is_batchnorm=True
               ):
    super(conv2DBatchNorm, self).__init__()

    conv_mod = nn.Conv2d(
        int(in_channels),
        int(n_filters),
        kernel_size=k_size,
        padding=padding,
        stride=stride,
        bias=bias,
        dilation=dilation,
        )

    if is_batchnorm:
      self.cb_unit = nn.Sequential(conv_mod, nn.BatchNorm2d(int(n_filters)))
    else:
      self.cb_unit = nn.Sequential(conv_mod)

  def forward(self, inputs):
    outputs = self.cb_unit(inputs)
    return outputs


class conv2DBatchNormRelu(nn.Module):
  def __init__(self,
               in_channels,
               n_filters,
               k_size,
               stride,
               padding,
               bias=True,
               dilation=1,
               is_batchnorm=True
               ):
    super(conv2DBatchNormRelu, self).__init__()

    conv_mod = nn.Conv2d(
        int(in_channels),
        int(n_filters),
        kernel_size=k_size,
        padding=padding,
        stride=stride,
        bias=bias,
        dilation=dilation,
        )

    if is_batchnorm:
      self.cbr_unit = nn.Sequential(
          conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True)
          )
    else:
      self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

  def forward(self, inputs):
    outputs = self.cbr_unit(inputs)
    return outputs


class conv2DGroupNormRelu(nn.Module):
  def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, n_groups=16
               ):
    super(conv2DGroupNormRelu, self).__init__()

    conv_mod = nn.Conv2d(
        int(in_channels),
        int(n_filters),
        kernel_size=k_size,
        padding=padding,
        stride=stride,
        bias=bias,
        dilation=dilation,
        )

    self.cgr_unit = nn.Sequential(
        conv_mod, nn.GroupNorm(n_groups, int(n_filters)), nn.ReLU(inplace=True)
        )

  def forward(self, inputs):
    outputs = self.cgr_unit(inputs)
    return outputs
