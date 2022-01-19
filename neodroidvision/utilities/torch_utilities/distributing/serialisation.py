import pickle
from typing import Any, List

import torch

__all__ = ["to_byte_tensor", "serialise_byte_tensor", "deserialise_byte_tensor"]


def to_byte_tensor(data: Any, *, device: str = "cuda") -> torch.ByteTensor:
    """

    :param data:
    :param device:
    :return:
    """
    return torch.ByteTensor(
        torch.ByteStorage.from_buffer(
            pickle.dumps(data)  # gets a byte representation for the data
        )
    ).to(
        device
    )  # convert this byte string into a byte tensor


def serialise_byte_tensor(
    encoded_data: Any, data: Any, *, device: str = "cuda"
) -> None:
    """


    :param encoded_data:
    :param data:
    :return:
    """

    tensor = to_byte_tensor(data, device=device)

    s = tensor.numel()  # encoding: first byte is the size and then rest is the data
    assert s <= 255, "Can't encode data greater than 255 bytes"

    encoded_data[0] = s  # put the size in encoded_data
    encoded_data[1 : (s + 1)] = tensor  # put the encoded data in encoded_data


def deserialise_byte_tensor(size_list, tensor_list) -> List:
    """

    :param size_list:
    :param tensor_list:
    :return:
    """
    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        data_list.append(pickle.loads(tensor.cpu().numpy().tobytes()[:size]))

    return data_list
