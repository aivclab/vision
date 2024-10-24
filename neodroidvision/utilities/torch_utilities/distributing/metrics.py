#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 23/03/2020
           """

import typing

import torch
from draugr.writers import Writer
from torch import distributed

from neodroidvision.utilities.torch_utilities.distributing.distributing_utilities import (
    global_world_size,
)

__all__ = ["write_metrics_recursive", "reduce_loss_dict"]


def write_metrics_recursive(
    eval_result: typing.Mapping, prefix: str, summary_writer: Writer, global_step: int
) -> None:
    """

    :param eval_result:
    :param prefix:
    :param summary_writer:
    :param global_step:
    """
    for key in eval_result:
        value = eval_result[key]
        tag = f"{prefix}/{key}"
        if isinstance(value, typing.Mapping):
            write_metrics_recursive(value, tag, summary_writer, global_step)
        else:
            summary_writer.scalar(tag, value, step_i=global_step)


def reduce_loss_dict(loss_dict: dict) -> dict:
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction."""
    world_size = global_world_size()

    if world_size < 2:
        return loss_dict

    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        distributed.reduce(all_losses, dst=0)
        if distributed.get_rank() == 0:
            # only main process gets accumulated, so only divide by world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses
