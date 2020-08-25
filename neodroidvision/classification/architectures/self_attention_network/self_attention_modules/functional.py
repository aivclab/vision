from enum import Enum

from . import functions

__all__ = ["aggregation", "subtraction", "subtraction2"]


class PadModeEnum(Enum):
    zero_pad = 0
    ref_pad = 1


def aggregation(
    input,
    weight,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    pad_mode: PadModeEnum = PadModeEnum.ref_pad,
):
    """

  :param input:
  :type input:
  :param weight:
  :type weight:
  :param kernel_size:
  :type kernel_size:
  :param stride:
  :type stride:
  :param padding:
  :type padding:
  :param dilation:
  :type dilation:
  :param pad_mode:
  :type pad_mode:
  :return:
  :rtype:
  """
    assert (
        input.shape[0] == weight.shape[0]
        and (input.shape[1] % weight.shape[1] == 0)
        and pad_mode in [0, 1]
    )
    if input.is_cuda:
        if pad_mode == PadModeEnum.zero_pad:
            out = functions.aggregation_zeropad(
                input, weight, kernel_size, stride, padding, dilation
            )
        elif pad_mode == PadModeEnum.ref_pad:
            out = functions.aggregation_refpad(
                input, weight, kernel_size, stride, padding, dilation
            )
    else:
        raise NotImplementedError
    return out


def subtraction(
    input,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    pad_mode: PadModeEnum = PadModeEnum.ref_pad,
):
    """

  :param input:
  :type input:
  :param kernel_size:
  :type kernel_size:
  :param stride:
  :type stride:
  :param padding:
  :type padding:
  :param dilation:
  :type dilation:
  :param pad_mode:
  :type pad_mode:
  :return:
  :rtype:
  """
    assert input.dim() == 4 and pad_mode in [0, 1]
    if input.is_cuda:
        if pad_mode == PadModeEnum.zero_pad:
            out = functions.subtraction_zeropad(
                input, kernel_size, stride, padding, dilation
            )
        elif pad_mode == PadModeEnum.ref_pad:
            out = functions.subtraction_refpad(
                input, kernel_size, stride, padding, dilation
            )
    else:
        raise NotImplementedError
    return out


def subtraction2(
    input1,
    input2,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    pad_mode: PadModeEnum = PadModeEnum.ref_pad,
):
    """

  :param input1:
  :type input1:
  :param input2:
  :type input2:
  :param kernel_size:
  :type kernel_size:
  :param stride:
  :type stride:
  :param padding:
  :type padding:
  :param dilation:
  :type dilation:
  :param pad_mode:
  :type pad_mode:
  :return:
  :rtype:
  """
    assert input1.dim() == 4 and input2.dim() == 4 and pad_mode in [0, 1]
    if input1.is_cuda:
        if pad_mode == PadModeEnum.zero_pad:
            out = functions.subtraction2_zeropad(
                input1, input2, kernel_size, stride, padding, dilation
            )
        elif pad_mode == PadModeEnum.ref_pad:
            out = functions.subtraction2_refpad(
                input1, input2, kernel_size, stride, padding, dilation
            )
    else:
        raise NotImplementedError
    return out
