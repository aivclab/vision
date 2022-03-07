"""
A package of torchvision transforms
Add examples to the __main__ routine as necessary for visualization
"""

from collections.abc import Sequence

import numpy
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image


def roundtoint(x):
    return int(round(x))


def tuplemulti(a, b, op=None):
    """
    elementwise multiplication of two sequences
    a and b are two equally long sequences
    op is a post-multi operation, default None
    """
    if op is not None:
        return [op(x * y) for x, y in zip(a, b)]
    return [x * y for x, y in zip(a, b)]


class BaseTorchTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        retmode = "tensor"
        # determine input type and possibly convert to tensor
        if isinstance(img, (Image.Image, numpy.ndarray)):
            img = to_tensor(img)
            retmode = "image"
        assert isinstance(
            img, torch.Tensor
        ), f"input must be convertable to torch tensor, got {type(img)}"

        return img, retmode


class RandomRGBNoise(BaseTorchTransform):
    """
    Inserts random R, G, and/or B values in an image
    Ratio affects the total amount of RGB values affected, not the amount of pixels.
    """

    def __init__(self, ratio=0.1, val=(0, 1)):
        super().__init__()
        self.ratio = ratio
        self.vals = val if isinstance(val, Sequence) else (0, val)

    def forward(self, img):
        img, retmode = super().forward(img)

        # first, create a random array of values 0..1
        # anything under ratio is considered a "hit"
        mod = torch.FloatTensor(*img.shape).uniform_(0.0, 1.0)
        hit = mod < self.ratio
        # then, set img values in the "hit" spaces to a random value in range self.vals
        img[hit] = img[hit].uniform_(*self.vals)

        if retmode == "image":
            return to_pil_image(img)
        return img


class RandomFlipColor(BaseTorchTransform):
    """
    Randomly flips a colour channel in an image
    Each channel is individually evaluated against ratio (R)
    Hence, the chance of each channel (C) being affected is R^C
        and the chance of no channels being affected (N) is (1-R)^C
        and the chance of any channel(s) being affected is thus 1-N
    If you wish to calculate the R required for total risk T of
        any one or more channels being affected in a single forward,
        use this formula:
            R = 1 - (1 - T)^(1/C)
        For example, if your target T is 0.8 and there are 3 channels
            in an image, R = 1 - (1 - 0.8)^(1/3) = 0.415
    """

    def __init__(self, ratio=0.1):
        super().__init__()
        self.ratio = ratio

    def forward(self, img):
        img, retmode = super().forward(img)

        # colours are assumed to be the first channel
        # anything under ratio is considered a "hit"
        mod = torch.FloatTensor(img.shape[0]).uniform_(0.0, 1.0)
        hit = mod < self.ratio
        # then, flip values in the "hit" channels
        img[hit] = 1.0 - img[hit]
        if retmode == "image":
            return to_pil_image(img)
        return img


class Black2RGB(BaseTorchTransform):
    """
    Inserts random RGB values in an image at specific dark colours
    Used to, for example, convert black writing to colored text
    """

    def __init__(self, threshold=0.25, vals=(0.0, 1.0), constant=True):
        """
        :param threshold: indicates the maximum value on any colour channel that causes a colour flip
        :param vals: possible values of R/G/B
        :param constant: whether to use a "constant" (read: gradient) colour or individual RGB values for each pixel
        """
        super().__init__()
        self.threshold = threshold
        self.constant = constant
        self.vals = vals if isinstance(vals, Sequence) else (0.0, vals)

    def forward(self, img):
        img, retmode = super().forward(img)

        # check if all values in pixel are below threshold
        hit = (img[:3] < self.threshold).all(dim=0)
        if self.constant:
            # norm is magnitude of vector
            a = img[:3, hit].norm(dim=0)
            # convert to 0..1 clamped at threshold
            a = a.clamp(0, self.threshold) / self.threshold
            col = torch.zeros((3, 1)).uniform_(*self.vals[:2])
            img[:3, hit] = img[:3, hit] * a + col * (1 - a)
        else:
            img[:3, hit] = img[:3, hit].uniform_(*self.vals)

        if retmode == "image":
            return to_pil_image(img)
        return img
