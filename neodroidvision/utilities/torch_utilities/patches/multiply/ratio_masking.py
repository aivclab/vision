import numpy
import torch

__all__ = ["RatioMaskGenerator"]

from torch import nn
from torch.nn.functional import fold


class RatioMaskGenerator(nn.Module):
    def __init__(self, patch_size, mask_ratio):
        super().__init__()
        if not isinstance(patch_size, tuple):
            patch_size = (patch_size,) * 2

        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

    def __repr__(self):
        return f"Maks: total patches {self.num_patches}, mask patches {self.num_mask}"

    def __call__(self, x):
        height, width = x.shape[-2:]
        num_patches = height // self.patch_size[0] * width // self.patch_size[1]
        num_mask = int(self.mask_ratio * num_patches)
        mask = numpy.vstack(
            [
                numpy.zeros((num_patches - num_mask, *self.patch_size)),
                numpy.ones((num_mask, *self.patch_size)),
            ]
        )
        numpy.random.shuffle(mask)
        mask = torch.from_numpy(mask)
        print(mask.shape, x.shape)
        resized = fold(
            mask, x.shape[-2:], self.patch_size, stride=self.patch_size, padding=0
        )
        return x * resized


if __name__ == "__main__":

    def asidj():
        from cv2 import circle
        import numpy

        shuffle = RatioMaskGenerator(20, 0.8)
        x_ = torch.randn(100, 100, 3).numpy() * 255  # batch, c, h, w, d

        x_ = circle(x_, (50, 50), 40, (200, 160, 120), -1).astype(numpy.uint8)

        from matplotlib import pyplot

        pyplot.imshow(x_)
        pyplot.show()
        x_ = torch.FloatTensor(x_).permute(2, 0, 1).contiguous().unsqueeze(0)
        shuffled = shuffle(x_)
        pyplot.imshow(shuffled.squeeze(0).permute(1, 2, 0).to(dtype=torch.int))
        pyplot.show()

    asidj()
