#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 5/5/22
           """

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Code is copy-pasted exactly as in torch.utils.data.distributed.
# FIXME remove this once c10d fixes the bug it has
import math
from typing import Sized

import torch
from torch import distributed
from torch.utils.data.sampler import Sampler

__all__ = ["DistributedSampler"]


# from torchvision.datasets.samplers import DistributedSampler
class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
    Dataset is assumed to be of constant size.
    Arguments:
    dataset: Dataset used for sampling.
    num_replicas (optional): Number of processes participating in
    distributed training.
    rank (optional): Rank of the current process within num_replicas."""

    def __init__(
        self, dataset: Sized, num_replicas: int = None, rank=None, shuffle: bool = True
    ):
        """


        :param dataset:
        :param num_replicas:
        :param rank:
        :param shuffle:"""
        if num_replicas is None:
            if not distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = distributed.get_world_size()
        if rank is None:
            if not distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        """

        :param epoch:
        :type epoch:
        """
        self.epoch = epoch
