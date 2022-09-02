#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/03/2020
           """

import cv2
import torch
from draugr.opencv_utilities import frame_generator
from draugr.torch_utilities import (
    TorchDeviceSession,
    TorchEvalSession,
    global_torch_device,
)
from draugr.visualisation import progress_bar
from matplotlib.pyplot import show
from neodroidvision import PROJECT_APP_PATH
from torchvision import transforms


@torch.no_grad()
def run_seg_traced_webcam_demo():
    """

    :return:
    :rtype:"""

    import torch
    import io

    load_path = (
        PROJECT_APP_PATH.user_data / "penn_fudan_segmentation" / "seg_skip_fis"
    ).with_suffix(".traced")
    # print(load_path)
    # torch.jit.load(str(load_path))

    with open(str(load_path), "rb") as f:  # Load ScriptModule from io.BytesIO object
        buffer = io.BytesIO(f.read())

    model = torch.jit.load(buffer)  # Load all tensors to the original device

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    from matplotlib.pyplot import imshow

    with TorchDeviceSession(device=global_torch_device("cpu"), model=model):
        with TorchEvalSession(model):
            for image in progress_bar(frame_generator(cv2.VideoCapture(0))):
                result = model(transform(image).unsqueeze(0).to(global_torch_device()))[
                    0
                ]

                imshow(result[0][0].numpy(), vmin=0.0, vmax=1.0)
                show()


def main():
    """description"""
    global_torch_device(override=global_torch_device("cpu"))

    run_seg_traced_webcam_demo()


if __name__ == "__main__":
    main()
