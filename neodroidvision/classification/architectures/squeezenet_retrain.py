import torch
import torchvision
from torch import nn

from neodroidvision.classification.architectures.retrain_utilities import set_all_parameter_requires_grad


def squeezenet_retrain(num_classes, pretrained=True, train_only_last_layer=False):
  model = torchvision.models.squeezenet1_1(pretrained=pretrained)
  if train_only_last_layer:
    set_all_parameter_requires_grad(model)

  model.num_classes = num_classes
  model.classifier[1] = torch.nn.Conv2d(512,
                                        num_classes,
                                        kernel_size=(1, 1),
                                        stride=(1, 1))

  params_to_update = []
  for name, param in model.named_parameters():
    if param.requires_grad == True:
      params_to_update.append(param)

  return model, params_to_update
