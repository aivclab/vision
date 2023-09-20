#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 27/06/2020
           """

from neodroidvision.data.classification.imagenet.imagenet_2012 import ImageNet2012
from pathlib import Path
from warg import NOD

SAN_CONFIG = NOD(
    dataset_type=ImageNet2012,
    dataset_path=Path.home() / "Data" / "Datasets" / "ILSVRC2012",
    arch="san",
    self_attention_type=0,
    layers=[2, 1, 2, 4, 1],
    kernels=[3, 7, 7, 7, 7],
    ignore_label=2000,
    base_lr=0.1,
    epochs=100,
    start_epoch=0,
    step_epochs=[30, 60, 90],
    label_smoothing=0.1,
    scheduler="cosine",
    momentum=0.9,
    weight_decay=0.0001,
    manual_seed=None,
    print_freq=10,
    save_freq=1,
    train_gpu=[0, 1, 2, 3, 4, 5, 6, 7],
    workers=32,  # data loader workers
    batch_size=256,  # batch size for training
    batch_size_val=128,  # batch size for validation during training, memory and speed tradeoff
    batch_size_test=10,  # 100,
    evaluate=True,
    # evaluate on validation set, extra gpu memory needed and small batch_size_val is recommend
    dist_url="tcp://127.0.0.1:6789",
    dist_backend="nccl",
    multiprocessing_distributed=True,
    world_size=1,
    rank=0,
    test_gpu=[0],
    test_workers=10,
    mixup_alpha=None,  #
    model_path=None,  #
    save_path=None,  #
    weight=None,  # path to initial weight (default=none)
    resume=None,  # path to latest checkpoint (default=none)
)
