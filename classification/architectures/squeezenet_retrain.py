import torchvision
from torch import nn


def set_parameter_requires_grad(model, feature_extracting):
  if feature_extracting:
    for param in model.parameters():
      param.requires_grad = False


def squeezenet_retrain(num_classes, train_only_last_layer=True):
  # model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
  model = torchvision.models.squeezenet1_1(pretrained=True)
  set_parameter_requires_grad(model, train_only_last_layer)

  model.num_classes = num_classes
  model.classifier = nn.Sequential(nn.Dropout(p=0.5),
                                   nn.Conv2d(512, num_classes, kernel_size=1),
                                   nn.ReLU(inplace=True),
                                   nn.AvgPool2d(13, stride=1),
                                   )

  # num_ftrs = model.fc.in_features
  # model.fc = nn.Sequential(torch.nn.Linear(num_ftrs, num_classes),torch.nn.Softmax())
  # model.fc = torch.nn.Linear(num_ftrs, num_classes) ## USE this when using old model

  params_to_update = model.parameters()
  # print("Params to learn:")
  if train_only_last_layer:
    params_to_update = []
    for name, param in model.named_parameters():
      if param.requires_grad == True:
        params_to_update.append(param)
        # print("\t", name)
  else:
    for name, param in model.named_parameters():
      if param.requires_grad == True:
        pass
        # print("\t", name)

  return model, params_to_update
