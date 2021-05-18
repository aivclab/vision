#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from PIL import Image
from neodroid.environments.droid_environment import connect_dict
from torch.utils.data import Dataset
from torchvision.transforms import transforms

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 10/11/2019
           """


class NeodroidCameraObservationDataset(Dataset):
    """ """

    # mean = numpy.array([0.485, 0.456, 0.406])
    # std = numpy.array([0.229, 0.224, 0.225])

    inverse_transform = transforms.Compose(
        [
            # transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist()),
            transforms.ToPILImage()
        ]
    )

    def __init__(self, split: str = "train", resize_s=256, size=10e9):
        """
        :type resize_s: int or tuple(w,h)
        :param dataset_path: dataset directory
        :param split: train, valid, test"""

        t = []

        if split == "train":
            t.extend(
                [
                    transforms.RandomResizedCrop(resize_s),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        else:
            t.extend([transforms.Resize(resize_s), transforms.CenterCrop(resize_s)])

        t.extend(
            [
                transforms.ToTensor()
                # transforms.Normalize(self.mean, self.std)
            ]
        )

        self.trans = transforms.Compose(t)
        self.env = connect_dict()
        self.env_iter = iter(connect_dict())
        self.size = size

    def __len__(self):
        return int(self.size)

    def __getitem__(self, index):
        if not self.env.is_connected:
            self.env = connect_dict()
            self.env_iter = iter(self.env)

        state = next(self.env_iter)
        state = state[list(state.keys())[0]]
        img = state._sensor("RGB").value
        label = state._sensor("Class").value

        img = self.trans(Image.fromarray(img, "RGBA"))

        return img, label


if __name__ == "__main__":

    import tqdm
    import torch
    from matplotlib import pyplot

    batch_size = 32

    dateset = NeodroidCameraObservationDataset()

    test_loader = torch.utils.data.DataLoader(
        dateset, batch_size=batch_size, shuffle=False
    )

    for batch_idx, (imgs, label) in tqdm.tqdm(
        enumerate(test_loader),
        total=len(test_loader),
        desc="Bro",
        ncols=80,
        leave=False,
    ):
        print(label[0])
        pyplot.imshow(dateset.inverse_transform(imgs[0]))
        # pyplot.imshow(imgs)
        pyplot.show()
        break
