from torch import nn
from typing import Any


class SSDBackbone(nn.Module):
    """ """

    def __init__(self, IMAGE_SIZE: Any):
        super().__init__()

    def reset_parameters(self):
        """ """
        for m in self.extras.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
