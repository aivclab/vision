from typing import Iterable, Tuple

from draugr.torch_utilities import NamedTensorTuple
from torch.utils.data.dataloader import default_collate

__all__ = ["BatchCollator"]


class BatchCollator:
    """ """

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
            targets = NamedTensorTuple(**targets)

        else:
            targets = None

        return images, targets, img_ids
