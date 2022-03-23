#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 23/03/2020
           """

import os
import sys
from pathlib import Path

import torch

from neodroidvision.utilities.torch_utilities.distributing.distributing_utilities import (
    is_main_process,
    synchronise_torch_barrier,
)

try:
    from torch.hub import download_url_to_file
    from torch.hub import urlparse
    from torch.hub import HASH_REGEX
except ImportError:
    from torch.utils.model_zoo import download_url_to_file
    from torch.utils.model_zoo import urlparse
    from torch.utils.model_zoo import HASH_REGEX

__all__ = ["custom_cache_url", "load_state_dict_from_url"]


# very similar to https://github.com/pytorch/pytorch/blob/master/torch/utils/model_zoo.py
# but with a few improvements and modifications
def custom_cache_url(url: str, model_dir: Path = None, progress: bool = True) -> Path:
    r"""Loads the Torch serialized object at the given URL.
    If the object is already present in `model_dir`, it's deserialized and
    returned. The filename part of the URL should follow the naming convention
    ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
    digits of the SHA256 hash of the contents of the file. The hash is used to
    ensure unique names and to verify the contents of the file.
    The default value of `model_dir` is ``$TORCH_HOME/models`` where
    ``$TORCH_HOME`` defaults to ``~/.torch``. The default directory can be
    overridden with the ``$TORCH_MODEL_ZOO`` environment variable.
    Args:
    url (string): URL of the object to download
    model_dir (string, optional): directory in which to save the object
    progress (bool, optional): whether or not to display a progress bar to stderr
    Example:
    >>> cached_file = maskrcnn_benchmark.utils.model_zoo.custom_cache_url(
    'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')
    """
    if model_dir is None:
        model_dir = os.getenv(
            "TORCH_MODEL_ZOO",
            Path(os.path.expanduser(os.getenv("TORCH_HOME", "~/.torch"))) / "models",
        )
    if not model_dir.exists():
        model_dir.mkdir(parents=True)
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if filename == "model_final.pkl":
        # workaround as pre-trained Caffe2 models from Detectron have all the same filename
        # so make the full path the filename by replacing / with _
        filename = parts.path.replace("/", "_")
    cached_file = model_dir / filename
    if not cached_file.exists() and is_main_process():
        sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')
        hash_prefix = HASH_REGEX.search(filename)
        if hash_prefix is not None:
            hash_prefix = hash_prefix.group(1)
            # workaround: Caffe2 models don't have a hash, but follow the R-50 convention,
            # which matches the hash PyTorch uses. So we skip the hash matching
            # if the hash_prefix is less than 6 characters
            if len(hash_prefix) < 6:
                hash_prefix = None
        download_url_to_file(url, str(cached_file), hash_prefix, progress=progress)
    synchronise_torch_barrier()
    return cached_file


def load_state_dict_from_url(url, map_location="cpu"):
    """

    Args:
      url:
      map_location:

    Returns:

    """
    return torch.load(custom_cache_url(url), map_location=map_location)
