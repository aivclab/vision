import torch

__all__ = ["mask_patches"]


def mask_patches(x, prob):
    """

    :param x:
    :type x:
    :param prob:
    :type prob:
    :return:
    :rtype:
    """
    prob = torch.randn(x.shape[:2]) < prob
    x[prob] = torch.zeros(x.shape[2:], dtype=torch.int)
    return x
