import torch

__all__ = ["StochasticMaskGenerator", "RatioMaskGenerator"]

from torch import nn
from torch.nn.functional import unfold, fold


class StochasticMaskGenerator(nn.Module):
    def __init__(self, patch_size, prob):
        super().__init__()
        self.patch_size = patch_size
        self.prob = prob

    def __call__(self, x):
        unfolded = unfold(
            x, kernel_size=self.patch_size, stride=self.patch_size, padding=0
        )
        unfolded[..., torch.randn(unfolded.shape[-1]) < self.prob] = 0
        folded = fold(
            unfolded,
            x.shape[-2:],
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
        )
        return folded


class RatioMaskGenerator(nn.Module):
    def __init__(self, patch_size, ratio):
        super().__init__()
        self.patch_size = patch_size
        self.ratio = ratio

    def __call__(self, x):
        unfolded = unfold(
            x, kernel_size=self.patch_size, stride=self.patch_size, padding=0
        )
        masked_num = int(unfolded.shape[-1] * self.ratio)
        mask = torch.hstack(
            (torch.ones(masked_num), torch.zeros(unfolded.shape[-1] - masked_num))
        ).to(dtype=torch.bool)
        index = torch.randperm(mask.shape[0])
        unfolded[..., mask[index]] = 0
        folded = fold(
            unfolded,
            x.shape[-2:],
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
        )
        return folded


if __name__ == "__main__":

    def asidj():
        from cv2 import circle
        import numpy

        shuffle = StochasticMaskGenerator(16, 0.8)
        x_ = torch.randn(100, 100, 3).numpy() * 255  # batch, c, h, w, d

        x_ = circle(x_, (50, 50), 40, (200, 160, 120), -1).astype(numpy.uint8)

        from matplotlib import pyplot

        pyplot.imshow(x_)
        pyplot.show()
        x_ = torch.FloatTensor(x_).permute(2, 0, 1).contiguous().unsqueeze(0)
        shuffled = shuffle(x_)
        pyplot.imshow(shuffled.squeeze(0).permute(1, 2, 0).to(dtype=torch.int))
        pyplot.show()

    def asid22j():
        from cv2 import circle
        import numpy

        shuffle = RatioMaskGenerator(16, 0.8)
        x_ = torch.randn(100, 100, 3).numpy() * 255  # batch, c, h, w, d

        x_ = circle(x_, (50, 50), 40, (200, 160, 120), -1).astype(numpy.uint8)

        from matplotlib import pyplot

        pyplot.imshow(x_)
        pyplot.show()
        x_ = torch.FloatTensor(x_).permute(2, 0, 1).contiguous().unsqueeze(0)
        shuffled = shuffle(x_)
        pyplot.imshow(shuffled.squeeze(0).permute(1, 2, 0).to(dtype=torch.int))
        pyplot.show()

    # asidj()
    asid22j()
