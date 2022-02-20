#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 13/11/2019
           """
__all__ = ["other_retrain"]


def other_retrain(
    arch: str, model: torch.nn.Module, num_classes: int
) -> torch.nn.Module:
    """
    Inplace op but returns the model anyway
    """
    if arch.startswith("alexnet"):
        model.classifier[6] = torch.nn.Linear(
            model.classifier[6].in_features, num_classes
        )
        print(f"=> reshaped AlexNet classifier layer with: {str(model.classifier[6])}")

    elif arch.startswith("vgg"):
        model.classifier[6] = torch.nn.Linear(
            model.classifier[6].in_features, num_classes
        )
        print(f"=> reshaped VGG classifier layer with: {str(model.classifier[6])}")

    elif arch.startswith("densenet"):
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
        print(f"=> reshaped DenseNet classifier layer with: {str(model.classifier)}")

    elif arch.startswith("inception"):
        model.AuxLogits.fc = torch.nn.Linear(
            model.AuxLogits.fc.in_features, num_classes
        )
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model
