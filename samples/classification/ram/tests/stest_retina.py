from functools import reduce

import numpy
import torch
from draugr.visualisation import (
    np_array_to_pil_img,
    pil_img_to_np_array,
    pil_merge_images,
)
from matplotlib import pyplot

from samples.classification.ram.architecture.ram_modules import GlimpseSensor
from samples.classification.ram.ram_params import get_ram_config


def main():
    """ """
    data_dir = get_ram_config()["data_dir"]

    # load images
    imgs = []
    paths = [data_dir / "lenna.jpg", data_dir / "cat.jpg"]
    for i in range(len(paths)):
        img = pil_img_to_np_array(paths[i], desired_size=(512, 512), expand=True)
        imgs.append(torch.from_numpy(img))
    imgs = torch.cat(imgs).permute(0, 3, 1, 2)

    # loc = torch.Tensor(2, 2).uniform_(-1, 1)
    loc = torch.from_numpy(numpy.array([[0.0, 0.0], [0.0, 0.0]]))

    num_patches = 5
    scale = 2
    patch_size = 10

    ret = GlimpseSensor.Retina(
        size_first_patch=patch_size,
        num_patches_per_glimpse=num_patches,
        scale_factor_suc=scale,
    )
    glimpse = ret.foveate(imgs, loc).data.numpy()

    glimpse = numpy.reshape(glimpse, [2, num_patches, 3, patch_size, patch_size])
    glimpse = numpy.transpose(glimpse, [0, 1, 3, 4, 2])

    merged = []
    for i in range(len(glimpse)):
        g = glimpse[i]
        g = list(g)
        g = [np_array_to_pil_img(l) for l in g]
        res = reduce(pil_merge_images, list(g))
        merged.append(res)

    merged = [numpy.asarray(l, dtype="float32") / 255.0 for l in merged]

    fig, axs = pyplot.subplots(nrows=2, ncols=1)
    for i, ax in enumerate(axs.flat):
        axs[i].imshow(merged[i])
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)
    pyplot.show()


if __name__ == "__main__":
    main()
