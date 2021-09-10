#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

        Gradient-weighted Class Activation Mapping

           Created on 14-02-2021
           """

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy
import torch
from draugr.torch_utilities import GuidedBackPropReLUModel
from torchvision import models

from neodroidvision.utilities.processing import (
    overlay_cam_on_image,
    preprocess_image,
    unstandardise_image,
)


class GradientClassActivationMapping:
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
                self.gradients.append(grad)

            def __call__(self, x):
                outputs = []
                self.gradients = []
                for name, module in self.model._modules.items():
                    x = module(x)
                    if name in self.target_layers:
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
        self.cuda = use_cuda
        if self.cuda:
            self.model = model._use_cuda()

        self.extractor = GradientClassActivationMapping.ModelOutputs(
            self.model, self.feature_module, target_layer_names
        )

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img._use_cuda()

        features, output = self.extractor(input_img)

        if target_category == None:
            target_category = numpy.argmax(output.cpu().data.numpy())

        one_hot = numpy.zeros((1, output.size()[-1]), dtype=numpy.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot._use_cuda()

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


if __name__ == "__main__":
    """Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    """

    from apppath import ensure_existence

    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--use-cuda",
            action="store_true",
            default=False,
            help="Use NVIDIA GPU acceleration",
        )
        parser.add_argument(
            "--image-path",
            type=str,
            default=str(
                Path.home()
                / "Data"
                / "Vision"
                / "this_is_a_test"
                / "this_is_a_smaller_test.png"
            ),
            help="Input image path",
        )
        args = parser.parse_args()
        args.use_cuda = args.use_cuda and torch.cuda.is_available()
        if args.use_cuda:
            print("Using GPU for acceleration")
        else:
            print("Using CPU for computation")

        return args

    args = get_args()

    model = models.resnet50(pretrained=True)
    grad_cam = GradientClassActivationMapping(
        model=model,
        feature_module=model.layer4,
        target_layer_names=["2"],
        use_cuda=args.use_cuda,
    )

    img = cv2.imread(args.image_path, 1)
    img = numpy.float32(img) / 255
    # Opencv loads as BGR:
    img = img[:, :, ::-1]
    input_img = preprocess_image(img)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = (518,)
    grayscale_cam = grad_cam(input_img, target_category)

    grayscale_cam = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))
    cam = overlay_cam_on_image(img, grayscale_cam)

    gb_model = GuidedBackPropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input_img, target_category=target_category)
    gb = gb.transpose((1, 2, 0))

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = unstandardise_image(cam_mask * gb)
    gb = unstandardise_image(gb)

    exclude = ensure_existence(Path.cwd() / "exclude")

    cv2.imwrite(str(exclude / "cam.jpg"), cam)
    cv2.imwrite(str(exclude / "gb.jpg"), gb)
    cv2.imwrite(str(exclude / "cam_gb.jpg"), cam_gb)
