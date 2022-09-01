#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import torch
from PIL import Image
from warg import ensure_existence
from draugr.numpy_utilities import SplitEnum, SplitIndexer
from draugr.random_utilities import seed_stack
from draugr.torch_utilities import (
    TensorBoardPytorchWriter,
    TorchEvalSession,
    TorchTrainSession,
    global_torch_device,
    load_model,
    save_model,
    trainable_parameters,
)
from draugr.tqdm_utilities import progress_bar
from neodroidvision import PROJECT_APP_PATH
from neodroidvision.data.mixed import PennFudanDataset
from neodroidvision.detection.two_stage.mask_rcnn.architecture import (
    get_pretrained_instance_segmentation_maskrcnn,
)
from neodroidvision.detection.two_stage.mask_rcnn.maskrcnn_engine import (
    maskrcnn_evaluate,
    maskrcnn_train_single_epoch,
)
from torch.utils.data import DataLoader, Subset
from warg import GDKC
from warg.functions import collate_first_dim

__author__ = "Christian Heider Nielsen"
__doc__ = """ description """

if __name__ == "__main__":

    def main(
        dataset_root=Path.home() / "DataWin" / "Datasets",
        model_name=f"maskrcnn_pennfudanped",
    ):
        """description"""

        base_path = ensure_existence(PROJECT_APP_PATH.user_data / "maskrcnn")
        log_path = ensure_existence(PROJECT_APP_PATH.user_log / "maskrcnn")
        export_root = ensure_existence(base_path / "models")

        batch_size = 4
        num_epochs = 10
        optimiser_spec = GDKC(torch.optim.Adam, lr=3e-4)
        scheduler_spec = GDKC(
            torch.optim.lr_scheduler.StepLR,  # a learning rate scheduler which decreases the learning rate by
            step_size=3,  # 10x every 3 epochs
            gamma=0.1,
        )
        num_workers = 0
        seed_stack(3825)

        dataset = PennFudanDataset(
            dataset_root / "PennFudanPed",
            SplitEnum.training,
            return_variant=PennFudanDataset.PennFudanReturnVariantEnum.all,
        )
        dataset_validation = PennFudanDataset(
            dataset_root / "PennFudanPed",
            SplitEnum.validation,
            return_variant=PennFudanDataset.PennFudanReturnVariantEnum.all,
        )
        split = SplitIndexer(len(dataset), validation=0.3, testing=0)

        split_indices = torch.randperm(split.total_num).tolist()

        data_loader = DataLoader(
            Subset(dataset, split_indices[: -split.validation_num]),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_first_dim,
        )

        data_loader_val = DataLoader(
            Subset(dataset_validation, split_indices[-split.validation_num :]),
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_first_dim,
        )

        model = get_pretrained_instance_segmentation_maskrcnn(dataset.response_channels)
        optimiser = optimiser_spec(trainable_parameters(model))
        lr_scheduler = scheduler_spec(optimiser)

        if True:
            candidate = load_model(
                model_name=model_name,
                model_directory=export_root,
                raise_on_failure=False,
            )
            if candidate:
                model = candidate

        if True:
            with TorchTrainSession(model):
                with TensorBoardPytorchWriter(log_path / model_name) as writer:
                    for epoch_i in progress_bar(
                        range(num_epochs), description="Epoch #"
                    ):
                        maskrcnn_train_single_epoch(
                            model=model,
                            optimiser=optimiser,
                            data_loader=data_loader,
                            writer=writer,
                        )
                        lr_scheduler.step()  # update the learning rate
                        maskrcnn_evaluate(
                            model, data_loader_val, writer=writer
                        )  # evaluate on the validation dataset
                        save_model(
                            model, model_name=model_name, save_directory=export_root
                        )

        if True:
            with TorchEvalSession(model):  # put the model in evaluation mode
                img, _ = dataset_validation[0]  # pick one image from the test set

                with torch.no_grad():
                    prediction = model([img.to(global_torch_device())])

                from matplotlib import pyplot

                pyplot.imshow(
                    Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
                )
                pyplot.show()

                import cv2

                pyplot.imshow(
                    Image.fromarray(
                        prediction[0]["masks"][0, 0].mul(255).byte().cpu().numpy()
                    )
                )
                pyplot.show()

                (boxes, labels, scores) = (
                    prediction[0]["boxes"].to("cpu").numpy(),
                    prediction[0]["labels"].to("cpu").numpy(),
                    torch.sigmoid(prediction[0]["scores"]).to("cpu").numpy(),
                )

                from draugr.opencv_utilities import draw_bounding_boxes
                from draugr.torch_utilities import quick_to_pil_image

                indices = scores > 0.1

                cv2.namedWindow(model_name, cv2.WINDOW_NORMAL)
                cv2.imshow(
                    model_name,
                    draw_bounding_boxes(
                        quick_to_pil_image(img),
                        boxes[indices],
                        labels=labels[indices],
                        scores=scores[indices],
                        # categories=categories,
                    ),
                )

                cv2.waitKey()

                main()
