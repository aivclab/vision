#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import cv2
import torch
from draugr.opencv_utilities import (
    frame_generator,
    draw_bounding_boxes,
    WindowFlagEnum,
    show_image,
)
from draugr.random_utilities import seed_stack
from draugr.torch_utilities import (
    global_torch_device,
    TorchEvalSession,
    to_tensor_generator,
    uint_hwc_to_chw_float_tensor,
)
from draugr.torch_utilities import load_model
from draugr.torch_utilities.images.conversion import quick_to_pil_image
from tqdm import tqdm

from neodroidvision import PROJECT_APP_PATH
from neodroidvision.data.mixed import PennFudanDataset
from neodroidvision.detection.two_stage.mask_rcnn.architecture import (
    get_pretrained_instance_segmentation_maskrcnn,
)

__author__ = "Christian Heider Nielsen"
__doc__ = ""

if __name__ == "__main__":

    def main(model_name: str = "maskrcnn_pennfudanped", score_threshold=0.55):
        """

        Args:
          model_name:
          score_threshold:
        """
        base_path = PROJECT_APP_PATH.user_data / "maskrcnn"
        dataset_root = Path.home() / "Data"

        seed_stack(3825)

        dataset = (
            PennFudanDataset  # (dataset_root / "PennFudanPed", SplitEnum.training)
        )
        categories = dataset.categories

        if True:
            model = load_model(
                model_name=model_name, model_directory=base_path / "models"
            )
        else:
            model = get_pretrained_instance_segmentation_maskrcnn(
                dataset.response_channels
            )

        model.to(global_torch_device())
        cpu_device = torch.device("cpu")

        with torch.no_grad():
            with TorchEvalSession(model):
                for image in tqdm(
                    to_tensor_generator(
                        frame_generator(cv2.VideoCapture(0)),
                        device=global_torch_device(),
                    )
                ):
                    prediction = model(
                        # torch_vision_normalize_batch_nchw(
                        uint_hwc_to_chw_float_tensor(image).unsqueeze(0)
                        #    )
                    )[0]

                    (boxes, labels, scores) = (
                        prediction["boxes"].to(cpu_device).numpy(),
                        prediction["labels"].to(cpu_device).numpy(),
                        torch.sigmoid(prediction["scores"]).to(cpu_device).numpy(),
                    )

                    indices = scores > score_threshold

                    if show_image(
                        draw_bounding_boxes(
                            quick_to_pil_image(image),
                            boxes[indices],
                            labels=labels[indices],
                            scores=scores[indices],
                            categories=categories,
                        ),
                        model_name,
                        wait=True,
                    ):
                        break  # esc to quit

    main()
