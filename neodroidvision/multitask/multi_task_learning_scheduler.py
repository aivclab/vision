#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 19-09-2021
           """

import torch


def cache_backbone_results(model: torch.Module) -> None:
    """
    Cache and freeze the backbone part of the model and only update heads
    """
    pass


def switch_target_head(model: torch.Module) -> None:
    """
    Only single head at a time
    """
    pass


def common_head_training(model: torch.Module) -> None:
    """
    Train all heads a the same time
    """
    pass
