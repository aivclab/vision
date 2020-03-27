from typing import Iterable, Tuple

from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import BatchSampler

__all__ = ["LimitedBatchResampler", "BatchCollator"]


class LimitedBatchResampler(BatchSampler):
    """
Wraps a BatchSampler, re-sampling from it until
a specified number of iterations have been sampled
"""

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


class BatchCollator:
    def __init__(self, wrap: bool = True):
        self.wrap = wrap

    def __call__(self, batch: Iterable) -> Tuple:
        transposed_batch = list(zip(*batch))
        images = default_collate(transposed_batch[0])
        img_ids = default_collate(transposed_batch[2])

        if self.wrap:
            list_targets = transposed_batch[1]
            targets = {
                key: default_collate([d[key] for d in list_targets])
                for key in list_targets[0]
            }

        else:
            targets = None

        return images, targets, img_ids
