import torch

from torch import nn
from torch.nn.functional import unfold, fold


class ShufflePatches(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def __call__(self, x):
        unfolded = unfold(
            x, kernel_size=self.patch_size, stride=self.patch_size, padding=0
        )
        permuted = torch.cat(
            [b_[:, torch.randperm(b_.shape[-1])][None, ...] for b_ in unfolded], dim=0
        )
        folded = fold(
            permuted,
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

        shuffle = ShufflePatches(16)
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
