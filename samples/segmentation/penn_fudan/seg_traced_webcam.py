#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/03/2020
           """

import cv2
import torch
from matplotlib.pyplot import show
from torchvision import transforms
from tqdm import tqdm

from draugr.opencv_utilities import frame_generator
from draugr.torch_utilities import TorchEvalSession, global_torch_device
from draugr.torch_utilities.sessions.device_sessions import TorchDeviceSession


@torch.no_grad()
def run_seg_traced_webcam_demo():
    """

:param categories:
:type categories:
:param cfg:
:type cfg:
:param model_ckpt:
:type model_ckpt:
:param score_threshold:
:type score_threshold:
:param window_name:
:type window_name:
:return:
:rtype:
"""

    import torch
    import io

    torch.jit.load("seg_skip_fis.traced")

    with open(
        "seg_skip_fis.traced", "rb"
    ) as f:  # Load ScriptModule from io.BytesIO object
        buffer = io.BytesIO(f.read())

    model = torch.jit.load(buffer)  # Load all tensors to the original device

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    with TorchDeviceSession(
        device=global_torch_device(cuda_if_available=False), model=model
    ):
        with TorchEvalSession(model):
            for image in tqdm(frame_generator(cv2.VideoCapture(0))):
                result = model(transform(image).unsqueeze(0).to(global_torch_device()))[
                    0
                ]
                print(result)
                from matplotlib.pyplot import imshow

                imshow(result[0][0].numpy(), vmin=0.0, vmax=1.0)
                show()


def main():
    global_torch_device(override=global_torch_device(cuda_if_available=False))

    run_seg_traced_webcam_demo()


if __name__ == "__main__":
    main()
