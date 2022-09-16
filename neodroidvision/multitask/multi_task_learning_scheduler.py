#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 19-09-2021
           """

import torch

__all__ = ["cache_backbone_results", "switch_target_head", "common_head_training"]


def cache_backbone_results(model: torch.nn.Module) -> None:
    """
    Cache and freeze the backbone part of the model and only update heads
    """
    pass
    raise NotImplementedError


def switch_target_head(model: torch.nn.Module) -> None:
    """
    Only single head at a time
    """
    pass
    raise NotImplementedError


def common_head_training(model: torch.nn.Module) -> None:
    """
    Train all heads a the same time
    """
    pass
    raise NotImplementedError
