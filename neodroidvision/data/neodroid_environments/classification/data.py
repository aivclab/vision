#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple, Union

import neodroid
from PIL import Image
from draugr.multiprocessing_utilities import PooledQueueProcessor, PooledQueueTask
from draugr.torch_utilities import global_torch_device
from torch.utils.data import Dataset
from torchvision import transforms

__author__ = "Christian Heider Nielsen"

import torch

default_torch_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

default_torch_retransform = transforms.Compose([transforms.ToPILImage("RGB")])

__all__ = [
    "neodroid_env_classification_generator",
    "pooled_neodroid_env_classification_generator",
]


def neodroid_env_classification_generator(env, batch_size=64) -> Tuple:
    """

    :param env:
    :param batch_size:
    """
    while True:
        predictors = []
        class_responses = []
        while len(predictors) < batch_size:
            state = env.update()
            rgb_arr = state.sensor("RGB").value
            rgb_arr = Image.open(rgb_arr).convert("RGB")
            a_class = state.sensor("Class").value

            predictors.append(default_torch_transform(rgb_arr))
            class_responses.append(int(a_class))

        a = torch.stack(predictors).to(global_torch_device())
        b = torch.LongTensor(class_responses).to(global_torch_device())
        yield a, b


def pooled_neodroid_env_classification_generator(env, device, batch_size=64) -> Tuple:
    """

    :param env:
    :param device:
    :param batch_size:
    :return:
    """

    class FetchConvert(PooledQueueTask):
        """ """

        def __init__(
            self,
            env,
            device: Union[str, torch.device] = "cpu",
            batch_size: int = 64,
            *args,
            **kwargs
        ):
            """

            :param env:
            :param device:
            :param batch_size:
            :param args:
            :param kwargs:"""
            super().__init__(*args, **kwargs)

            self.env = env
            self.batch_size = batch_size
            self.device = device

        def call(self, *args, **kwargs) -> Tuple:
            """

            Args:
              *args:
              **kwargs:

            Returns:

            """
            predictors = []
            class_responses = []

            while len(predictors) < self.batch_size:
                state = self.env.update()
                rgb_arr = state.sensor("RGB").value
                rgb_arr = Image.open(rgb_arr).convert("RGB")
                a_class = state.sensor("Class").value

                predictors.append(default_torch_transform(rgb_arr))
                class_responses.append(int(a_class))

            return (
                torch.stack(predictors).to(self.device),
                torch.LongTensor(class_responses).to(self.device),
            )

    task = FetchConvert(env, device=device, batch_size=batch_size)

    processor = PooledQueueProcessor(
        task, fill_at_construction=True, max_queue_size=16, n_proc=None
    )

    for a in zip(processor):
        yield a


if __name__ == "__main__":

    def asdadsad():
        """ """
        neodroid_generator = neodroid_env_classification_generator(neodroid.connect())
        train_loader = torch.utils.data.DataLoader(
            dataset=neodroid_generator, batch_size=12, shuffle=True
        )
        for p, r in train_loader:
            print(r)

    asdadsad()
