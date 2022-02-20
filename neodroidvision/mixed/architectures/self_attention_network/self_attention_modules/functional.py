import torch

from . import functions

__all__ = ["aggregation", "subtraction", "subtraction2"]

from neodroidvision.mixed.architectures.self_attention_network.enums import PadModeEnum


def aggregation(
    input: torch.Tensor,
    weight: torch.Tensor,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    pad_mode: PadModeEnum = PadModeEnum.ref_pad,
) -> torch.Tensor:
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
    :rtype:"""
    assert (
        input.shape[0] == weight.shape[0]
        and (input.shape[1] % weight.shape[1] == 0)
        and pad_mode in PadModeEnum
    ), f"{input.shape, weight.shape, pad_mode}"
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
            raise NotImplementedError(pad_mode)
    else:
        raise NotImplementedError
    return out


def subtraction(
    input: torch.Tensor,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    pad_mode: PadModeEnum = PadModeEnum.ref_pad,
) -> torch.Tensor:
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
    :rtype:"""
    assert input.dim() == 4 and pad_mode in PadModeEnum, f"{input.shape, pad_mode}"
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
            raise NotImplementedError(pad_mode)
    else:
        raise NotImplementedError
    return out


def subtraction2(
    input1: torch.Tensor,
    input2: torch.Tensor,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    pad_mode: PadModeEnum = PadModeEnum.ref_pad,
) -> torch.Tensor:
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
    :rtype:"""
    assert (
        input1.dim() == 4 and input2.dim() == 4 and pad_mode in PadModeEnum
    ), f"{input1.shape, input2.shape, pad_mode}"
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
            raise NotImplementedError(pad_mode)
    else:
        raise NotImplementedError
    return out
