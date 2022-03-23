import glob
import random
from collections.abc import Sequence

import torch
import torch.nn.functional
from PIL import Image
from torch import ceil
from torch.nn.functional import interpolate
from torchvision.transforms.functional import to_tensor, to_pil_image
from utilities.torch_utilities.transforms.image_transforms import (
    BaseTorchTransform,
    tuplemulti,
)


class RandomBlock(BaseTorchTransform):
    """
    Randomly blocks out parts of an image
    Select mode from constant or rgb (random RGB values)
    Color is relevant in constant mode. Expects float (0..1) or a sequence of length C
    """

    def __init__(self, mode="constant", color=0, max_size=(1.0, 1.0)):
        super().__init__()
        self.mode = mode
        self.color = torch.tensor(color).unsqueeze(-1).unsqueeze(-1)

        # ensure max_size is a tuple/list
        if not isinstance(max_size, Sequence):
            max_size = (max_size, max_size)

        assert len(max_size) == 2, "max_size must have length 2 or be a scalar"
        assert (
            0.0 < max_size[0] <= 1.0 and 0.0 < max_size[1] <= 1.0
        ), f"max_size must be greater than zero and less than or equal to 1, got {max_size}"
        self.max_size = max_size

    def forward(self, img):
        img, retmode = super().forward(img)

        # generate random slice indices
        bh = torch.randint(int(round(img.shape[-2] * self.max_size[0])), (1,)).item()
        bw = torch.randint(int(round(img.shape[-1] * self.max_size[1])), (1,)).item()
        y = torch.randint(img.shape[-2] - bh, (1,)).item()
        x = torch.randint(img.shape[-1] - bw, (1,)).item()

        # constant uses color argument
        if self.mode == "constant":
            img[:, y : y + bh, x : x + bh] = self.color
        elif self.mode == "rgb":
            # assumes colour channels are channel 0
            c = torch.rand((img.shape[0], 1, 1))
            img[:, y : y + bh, x : x + bh] = c
        else:
            raise ValueError(
                f"self.mode is expected to be 'rgb' or 'constant', got '{self.mode}'"
            )

        if retmode == "image":
            return to_pil_image(img)
        return img


class BlockyMask(BaseTorchTransform):
    """
    Randomly blocks out parts of an image
    Select mode from constant or rgb (random RGB values)
    Color is relevant in constant mode. Expects float (0..1) or a sequence of length C
    """

    def __init__(self, mode="constant", color=0, max_size=(1.0, 1.0)):
        super().__init__()
        self.mode = mode
        self.color = torch.tensor(color).unsqueeze(-1).unsqueeze(-1)

        # ensure max_size is a tuple/list
        if not isinstance(max_size, Sequence):
            max_size = (max_size, max_size)

        assert len(max_size) == 2, "max_size must have length 2 or be a scalar"
        assert (
            0.0 < max_size[0] <= 1.0 and 0.0 < max_size[1] <= 1.0
        ), f"max_size must be greater than zero and less than or equal to 1, got {max_size}"
        self.max_size = max_size

    def forward(self, img):
        img, retmode = super().forward(img)

        # generate random slice indices
        bh = torch.randint(int(round(img.shape[-2] * self.max_size[0])), (1,)).item()
        bw = torch.randint(int(round(img.shape[-1] * self.max_size[1])), (1,)).item()
        y = torch.randint(img.shape[-2] - bh, (1,)).item()
        x = torch.randint(img.shape[-1] - bw, (1,)).item()

        # constant uses color argument
        if self.mode == "constant":
            img[:, y : y + bh, x : x + bh] = self.color
        elif self.mode == "rgb":
            # assumes colour channels are channel 0
            c = torch.rand((img.shape[0], 1, 1))
            img[:, y : y + bh, x : x + bh] = c
        else:
            raise ValueError(
                f"self.mode is expected to be 'rgb' or 'constant', got '{self.mode}'"
            )

        if retmode == "image":
            return to_pil_image(img)
        return img


class RandomImageBlocking(BaseTorchTransform):
    """
    Inserts random .png images on top of the input image
    Expects RGBA images, but will use RGB images without warning
    """

    def __init__(self, root_dir, apply=1.0, max_size=(1.0, 1.0), rotate=True):
        """
        :param root_dir: path to root directory of png images
        :param apply: float, a value between 0 and 1 resembles the chance of apllying the transform
        :param max_size: expects float or (float,float) - resembles the max size of the blocking image relative to the input image
        """
        super().__init__()
        self.images = glob.glob(root_dir + "**/*.png", recursive=True)
        self.apply = apply
        self.rotate = rotate

        if not isinstance(max_size, Sequence):
            max_size = (max_size, max_size)

        assert len(max_size) == 2, "max_size must have length 2 or be a scalar"
        assert (
            0.0 < max_size[0] <= 1.0 and 0.0 < max_size[1] <= 1.0
        ), f"max_size must be greater than zero and less than or equal to 1, got {max_size}"
        self.max_size = max_size

    def forward(self, img):
        """
        Randomly applies blocking images, likely PNGs, to input image
        :param img: input image, expects PIL.Image, numpy.ndarray, or torch.Tensor
        """

        # generates a single float value between 0..1 and evaluates against apply
        if torch.rand(1) > self.apply:
            return img

        img, retmode = super().forward(img)

        # load image
        block = Image.open(random.choice(self.images))
        mode = block.mode

        # should it be randomly rotated?
        if self.rotate:
            block.rotate(torch.randint(360, (1,)).item())

        block = to_tensor(block).unsqueeze(0)

        # scale is max_size * random float between 0..1
        # it is uniformly distributed between 0..max_size
        scale = tuplemulti(torch.rand(2).tolist(), self.max_size)
        block = interpolate(block, tuplemulti(img.shape[-2:], scale, op=ceil)).squeeze()

        # determine start positions
        y = torch.randint(img.shape[-2] - block.shape[-2], (1,)).item()
        x = torch.randint(img.shape[-1] - block.shape[-1], (1,)).item()

        # expected
        if mode == "RGBA":
            A = block[-1]
            block = block[:-1]
            img[:, y : y + block.shape[-2], x : x + block.shape[-1]] = (
                img[:, y : y + block.shape[-2], x : x + block.shape[-1]] * (1 - A)
                + block * A
            )
        # accepted
        elif mode == "RGB":
            img[:, y : y + block.shape[-2], x : x + block.shape[-1]] = block
        else:
            raise ValueError(
                f"Image.mode of blocking image is expected to be RGB(A), got {block.mode}"
            )

        if retmode == "image":
            return to_pil_image(img)
        return img
