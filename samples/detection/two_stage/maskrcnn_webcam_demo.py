#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import cv2
import numpy
import torch

__author__ = "Christian Heider Nielsen"
__doc__ = ""

from tqdm import tqdm

from draugr.opencv_utilities import frame_generator, draw_bouding_boxes
from draugr.torch_utilities import (
    global_torch_device,
    torch_seed,
    TorchEvalSession,
    to_tensor_generator,
)
from neodroidvision.data.datasets.supervised.segmentation import PennFudanDataset
from neodroidvision.data.datasets.supervised.splitting import Split
from neodroidvision.detection.two_stage.mask_rcnn.architecture import (
    get_pretrained_instance_segmentation_maskrcnn,
)

if __name__ == "__main__":

    def main(window_name: str = "SSD", score_threshold=0.7):
        dataset_root = Path("/home/heider/Data/Datasets")

        torch_seed(3825)

        dataset = PennFudanDataset(dataset_root / "PennFudanPed", Split.Training)
        categories = dataset.categories

        model = get_pretrained_instance_segmentation_maskrcnn(dataset.response_channels)
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

                    prediction = model([image])

                    (boxes, labels, scores) = (
                        prediction["boxes"].to(cpu_device).numpy(),
                        prediction["labels"].to(cpu_device).numpy(),
                        prediction["scores"].to(cpu_device).numpy(),
                    )

                    indices = scores > score_threshold

                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.imshow(
                        window_name,
                        draw_bouding_boxes(
                            image,
                            boxes[indices],
                            labels[indices],
                            scores[indices],
                            categories,
                        ).astype(numpy.uint8),
                    )

                    if cv2.waitKey(1) == 27:
                        break  # esc to quit

    main()
