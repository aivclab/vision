from typing import List, Tuple

import torch
import torchvision
from draugr.torch_utilities import (
    set_all_parameter_requires_grad,
    trainable_parameters,
)

__all__ = ["squeezenet_retrain"]

from torch.nn.parameter import Parameter
from torchvision.models import SqueezeNet


def squeezenet_retrain(
    num_classes: int, pretrained: bool = True, train_only_last_layer: bool = False
) -> Tuple[SqueezeNet, List[Parameter]]:
    """

    Args:
      num_classes:
      pretrained:
      train_only_last_layer:

    Returns:

    """
    model = torchvision.models.squeezenet1_1(pretrained=pretrained)
    if train_only_last_layer:
        set_all_parameter_requires_grad(model)

    model.num_categories = num_classes
    model.classifier[1] = torch.nn.Conv2d(
        512, num_classes, kernel_size=(1, 1), stride=(1, 1)
    )

    return model, trainable_parameters(model)
