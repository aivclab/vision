#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

        Gradient-weighted Class Activation Mapping

           Created on 14-02-2021
           """

from typing import Sequence

import cv2
import numpy
import torch

__all__ = ["GradientClassActivationMapping"]


class GradientClassActivationMapping:
    """ """

    class ModelOutputs:
        """Class for making a forward pass, and getting:
        1. The network output.
        2. Activations from intermeddiate targetted layers.
        3. Gradients from intermeddiate targetted layers."""

        class FeatureExtractor:
            """Class for extracting activations and
            registering gradients from targetted intermediate layers
            """

            def __init__(self, model, target_layers):
                self.model = model
                self.target_layers = target_layers
                self.gradients = []

            def save_gradient(self, grad):
                """

                Args:
                  grad:
                """
                self.gradients.append(grad)

            def __call__(self, x):
                outputs = []
                self.gradients = []
                for name, module in self.model._modules.items():
                    x = module(x)
                    # print(name)
                    if name in self.target_layers:
                        # print(f'registered {name}')
                        x.register_hook(self.save_gradient)
                        outputs += [x]
                return outputs, x

        def __init__(self, model, feature_module, target_layers):
            self.model = model
            self.feature_module = feature_module
            self.feature_extractor = (
                GradientClassActivationMapping.ModelOutputs.FeatureExtractor(
                    self.feature_module, target_layers
                )
            )

        def get_gradients(self):
            """

            Returns:

            """
            return self.feature_extractor.gradients

        def __call__(self, x):
            target_activations = []
            for name, module in self.model._modules.items():
                if module == self.feature_module:
                    target_activations, x = self.feature_extractor(x)
                elif "avgpool" in name.lower():
                    x = module(x)
                    x = x.view(x.size(0), -1)
                else:
                    x = module(x)

            return target_activations, x

    def __init__(
        self,
        model: torch.nn.Module,
        feature_module: torch.nn.Module,
        target_layer_names: Sequence,
        use_cuda: bool,
    ):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model = model.cuda()

        self.extractor = GradientClassActivationMapping.ModelOutputs(
            self.model, self.feature_module, target_layer_names
        )

    def forward(self, input_img):
        """

        Args:
          input_img:

        Returns:

        """
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.use_cuda:
            input_img = input_img.cuda()

        features, output = self.extractor(input_img)

        if target_category == None:
            target_category = numpy.argmax(output.cpu().data.numpy())

        one_hot = numpy.zeros((1, output.size()[-1]), dtype=numpy.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.use_cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = numpy.mean(grads_val, axis=(2, 3))[0, :]
        cam = numpy.zeros(target.shape[1:], dtype=numpy.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = cv2.resize(numpy.maximum(cam, 0), input_img.shape[2:])
        cam -= numpy.min(cam)
        cam /= numpy.max(cam)
        return cam
