import sys

from samples.classification.ram.architecture.ram import RecurrentAttention
from samples.classification.ram.ram_params import get_ram_config

from draugr import pil_img_to_np_array

sys.path.append("..")

import torch

if __name__ == "__main__":

    config = get_ram_config()

    # load images
    imgs = []
    paths = [config["data_dir"] / "lenna.jpg", config["data_dir"] / "cat.jpg"]
    for i in range(len(paths)):
        img = pil_img_to_np_array(paths[i], desired_size=[512, 512], expand=True)
        imgs.append(torch.from_numpy(img))
    imgs = torch.cat(imgs).permute((0, 3, 1, 2))

    B, C, H, W = imgs.shape
    l_t_prev = torch.FloatTensor(B, 2).uniform_(-1, 1)
    h_t_prev = torch.zeros(B, 256)

    ram = RecurrentAttention(64, 3, 2, C, 128, 128, 0.11, 256, 10)
    h_t, l_t, _, _ = ram(imgs, l_t_prev, h_t_prev)

    assert h_t.shape == (B, 256)
    assert l_t.shape == (B, 2)
