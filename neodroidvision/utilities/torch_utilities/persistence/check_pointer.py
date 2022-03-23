#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 23/03/2020
           """

import logging
from pathlib import Path
from typing import Any

import torch
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel

from neodroidvision.utilities.torch_utilities.persistence.custom_model_caching import (
    custom_cache_url,
)

__all__ = ["CheckPointer"]


class CheckPointer:
    """"""

    _last_checkpoint_name = "last_checkpoint.txt"

    def __init__(
        self,
        model: Module,
        optimiser: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        save_dir: Path = Path.cwd(),
        save_to_disk: bool = None,
        logger: logging.Logger = None,
    ):
        """

        :param model:
        :type model:
        :param optimiser:
        :type optimiser:
        :param scheduler:
        :type scheduler:
        :param save_dir:
        :type save_dir:
        :param save_to_disk:
        :type save_to_disk:
        :param logger:
        :type logger:"""
        self.model = model
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        """

        Args:
          self:
          name:
          **kwargs:

        Returns:

        """
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        if isinstance(self.model, DistributedDataParallel):
            data["model"] = self.model.module.state_dict()
        else:
            data["model"] = self.model.state_dict()
        if self.optimiser is not None:
            data["optimiser"] = self.optimiser.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = self.save_dir / f"{name}.pth"
        self.logger.info(f"Saving checkpoint to {save_file}")
        torch.save(data, save_file)

        self.tag_last_checkpoint(save_file)

    def load(self, f: Path = None, use_latest=True):
        """

        Args:
          self:
          f:
          use_latest:

        Returns:

        """
        if f is None:
            return {}

        f = str(f)
        if (self.save_dir / self._last_checkpoint_name).exists() and use_latest:
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()

        if f is None or f == "" or f == "None":
            # no checkpoint could be found
            self.logger.info("No checkpoint found.")
            return {}

        self.logger.info(f"Loading checkpoint from {f}")
        checkpoint = self._load_file(f)
        model = self.model
        if isinstance(model, DistributedDataParallel):
            model = self.model.module

        model.load_state_dict(checkpoint.pop("model"))
        if "optimiser" in checkpoint and self.optimiser:
            self.logger.info(f"Loading optimiser from {f}")
            self.optimiser.load_state_dict(checkpoint.pop("optimiser"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info(f"Loading scheduler from {f}")
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def get_checkpoint_file(self) -> str:
        """

        Args:
          self:

        Returns:

        """
        try:
            with open(str(self.save_dir / self._last_checkpoint_name), "r") as f:
                last_saved = f.read().strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename) -> None:
        """

        Args:
          self:
          last_filename:
        """
        with open(str(self.save_dir / self._last_checkpoint_name), "w") as f:
            f.write(last_filename)

    def _load_file(self, f: str) -> Any:
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            f = custom_cache_url(f)
            self.logger.info(f"url {f} cached in {f}")
        return torch.load(f, map_location=torch.device("cpu"))
