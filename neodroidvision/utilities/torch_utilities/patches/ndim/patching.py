from itertools import count
from typing import Sequence

import numpy
import torch
import torch.nn.functional
from warg import split

"""
Proof of concept, slow
"""

__all__ = ["extract_patches", "combine_patches"]


def extract_patches(x, kernel, stride=None):
    """
    2d, 3d.. data

    :param x:
    :type x:
    :param kernel:
    :type kernel:
    :param stride:
    :type stride:
    :return:
    :rtype:
    """
    b, c, *dims = x.shape
    t_dim = len(x.shape)
    num_dim = len(dims)
    if not isinstance(kernel, Sequence):
        kernel = (kernel,) * num_dim
    if stride is None:
        stride = kernel
    assert len(dims) == len(kernel) == len(stride)

    for dim, k, s in zip(count(t_dim - num_dim), kernel, stride):
        x = x.unfold(dim, k, s)

    return (
        x.contiguous().view(x.size(0), c, -1, *kernel).transpose(1, 2).contiguous()
    )  # B, P, C, HW..


def combine_patches(x: torch.Tensor):
    b, p, c, *patch_size = x.shape
    x = x.transpose(1, 2).contiguous()  # B, C, P, HW..
    num_dim = len(patch_size)

    s = round(p ** (1 / num_dim))
    orik = (s * k for k in patch_size)
    r = (s,) * num_dim

    new_view = b, c, *r, *patch_size
    x = x.contiguous().view(new_view)

    l, h = split(list(range(2, len(x.shape))))
    new_permute = (b for a in zip(l, h) for b in a)
    x = x.permute(0, 1, *new_permute)

    # for i, d in zip(count(1), r):
    #  x = torch.nn.functional.fold(x, (i * s, i * s), d)

    return x.contiguous().view(x.size(0), c, *orik).contiguous()


if __name__ == "__main__":

    def suahd():
        show_2d = True
        patch_size = 8

        if show_2d:

            from cv2 import circle
            from matplotlib import pyplot

            x_ = torch.randn(100, 100, 3).numpy() * 255  # batch, c, h, w, d
            x_ = circle(x_, (50, 50), 40, (200, 160, 120), -1).astype(numpy.uint8)
            pyplot.imshow(x_)
            pyplot.show()
            x_ = torch.IntTensor(x_).permute(2, 0, 1).contiguous().unsqueeze(0)
        else:
            x_ = torch.randn(1, 3, 50, 50, 50)
        print(x_.shape)
        patches = extract_patches(x_.detach(), patch_size)

        combined = combine_patches(patches)

        print(x_.shape, patches.shape, combined.shape)

        s = combined.shape[2:]
        if show_2d:
            x_crop = x_[:, :, : s[0], : s[1]]
            print((combined == x_crop).all(), torch.sum(x_crop - combined))
        else:
            x_crop = x_[:, :, : s[0], : s[1], : s[2]]
            print((combined == x_crop).all(), torch.sum(x_crop - combined))

        if show_2d:
            from math import sqrt

            acc = len(patches[0])
            f, ax = pyplot.subplots(int(sqrt(acc)), int(sqrt(acc)))
            ax = [a_b for a in ax for a_b in a]
            for a, p in zip(ax, patches[0]):
                a.imshow(p.permute(1, 2, 0))
            pyplot.show()

            pyplot.imshow(combined[0].permute(1, 2, 0))
            pyplot.show()

    def suahsadd():
        from time import time

        patch_size = 8

        from sampling import mask_patches
        from cv2 import circle
        from matplotlib import pyplot

        x_ = torch.randn(100, 100, 3).numpy() * 255  # batch, c, h, w, d
        x_ = circle(x_, (50, 50), 40, (200, 160, 120), -1).astype(numpy.uint8)
        if False:
            pyplot.imshow(x_)
            pyplot.show()
        x_ = torch.IntTensor(x_).permute(2, 0, 1).contiguous().unsqueeze(0)

        t1 = time()
        patches = extract_patches(x_.detach(), patch_size)
        masked = mask_patches(patches, prob=0.6)
        combined = combine_patches(masked)
        print(time() - t1)

        if False:
            from math import sqrt

            acc = len(patches[0])
            f, ax = pyplot.subplots(int(sqrt(acc)), int(sqrt(acc)))
            ax = [a_b for a in ax for a_b in a]
            for a, p in zip(ax, patches[0]):
                a.imshow(p.permute(1, 2, 0))
            pyplot.show()

        pyplot.imshow(combined[0].permute(1, 2, 0))
        pyplot.show()

    suahsadd()
