import torch
from torchvision.transforms.functional import to_pil_image

from utilities.torch_utilities.patches.shuffle import extract_patches, combine_patches
from utilities.torch_utilities.transforms.image_transforms import BaseTorchTransform

__all__ = ["BlockyShuffle"]


def shuffle_patches(patches: torch.Tensor):
    patches


class BlockyShuffle(BaseTorchTransform):
    """
    Randomly blocks out parts of an image
    Select mode from constant or rgb (random RGB values)
    Color is relevant in constant mode. Expects float (0..1) or a sequence of length C
    """

    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, img):
        img, retmode = super().forward(img)

        img = img.unsqueeze(0)
        patches = extract_patches(img, self.patch_size)
        shuffled = shuffle_patches(patches)
        combined = combine_patches(shuffled)
        img = combined.squeeze(0)

        if retmode == "image":
            return to_pil_image(img)
        return img
