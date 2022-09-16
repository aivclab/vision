#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 5/5/22
           """

import datetime
import time
from collections import defaultdict, deque
from typing import Optional

import torch
import torch.utils.data
from torch import distributed

__all__ = ["SmoothedValue", "MetricLogger"]

from neodroidvision.utilities.torch_utilities.distributing.distributing_utilities import (
    is_distribution_ready,
)


class SmoothedValue(object):
    """
    Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt: Optional[str] = None):
        if fmt is None:
            self.fmt = "{median:.4f} ({global_avg:.4f})"
        else:
            self.fmt = fmt
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        """

        Args:
          value:
          n:
        """
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronise_between_processes_torch(self):
        """
        Warning: does not synchronize the deque!"""
        if not is_distribution_ready():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        distributed.barrier()
        distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        """

        Returns:

        """
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        """

        Returns:

        """
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        """

        Returns:

        """
        return self.total / self.count

    @property
    def max(self):
        """

        Returns:

        """
        return max(self.deque)

    @property
    def value(self):
        """

        Returns:

        """
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    """description"""

    MB = 1024.0 * 1024.0

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        """

        Args:
          **kwargs:
        """
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronise_meters_between_processes(self):
        """description"""
        for meter in self.meters.values():
            meter.synchronise_between_processes_torch()

    def add_meter(self, name, meter):
        """

        Args:
          name:
          meter:
        """
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """

        Args:
          iterable:
          print_freq:
          header:
        """
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = f":{str(len(str(len(iterable))))}d"
        log_msg = self.delimiter.join(
            [
                header,
                "[{0" + space_fmt + "}/{1}]",
                "eta: {eta}",
                "{meters}",
                "time: {time}",
                "data: {data}",
                "max mem: {memory:.0f}",
            ]
        )

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    log_msg.format(
                        i,
                        len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / self.MB,
                    )
                )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)"
        )
