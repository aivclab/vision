import torch.nn as nn


class SSDBackbone(nn.Module):
    def __init__(self, IMAGE_SIZE):
        super().__init__()

    def reset_parameters(self):
        for m in self.extras.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
