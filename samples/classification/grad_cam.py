#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

        Gradient-weighted Class Activation Mapping

           Created on 14-02-2021
           """

from pathlib import Path

import cv2
import numpy
from draugr.torch_utilities import GuidedBackPropReLUModel
from neodroidvision.utilities import (
    GradientClassActivationMapping,
    overlay_cam_on_image,
    preprocess_image,
    unstandardise_image,
)
from torchvision import models

if __name__ == "__main__":

    def main():
        """Makes a forward pass to find the category index with the highest score,
        and computes intermediate activations.
        """

        from apppath import ensure_existence

        use_cuda = True
        image_path = str(Path.home() / "Data" / "ok.png")
        model = models.resnet50(pretrained=True)
        grad_cam = GradientClassActivationMapping(
            model=model,
            feature_module=model.layer4,
            target_layer_names=["2"],
            use_cuda=use_cuda,
        )
        """
model = models.resnet18(pretrained=True)
#print(list(model.named_parameters()))
grad_cam = GradientClassActivationMapping(
model=model,
feature_module=model.layer4,
target_layer_names=["1"],
use_cuda=use_cuda,
)
"""
        img = cv2.imread(image_path, 1)
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

        gb_model = GuidedBackPropReLUModel(model=model, use_cuda=use_cuda)
        gb = gb_model(input_img, target_category=target_category)
        gb = gb.transpose((1, 2, 0))

        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        cam_gb = unstandardise_image(cam_mask * gb)
        gb = unstandardise_image(gb)

        exclude = ensure_existence(Path.cwd() / "exclude")

        cv2.imwrite(str(exclude / "cam.jpg"), cam)
        cv2.imwrite(str(exclude / "gb.jpg"), gb)
        cv2.imwrite(str(exclude / "cam_gb.jpg"), cam_gb)

    main()
