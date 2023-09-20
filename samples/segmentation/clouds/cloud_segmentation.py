#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """

import cv2
import numpy
import pandas
import torch
from draugr.numpy_utilities import SplitEnum, chw_to_hwc, float_chw_to_hwc_uint
from draugr.random_utilities import seed_stack
from draugr.torch_utilities import (
    TorchEvalSession,
    TorchTrainSession,
    global_torch_device,
)
from draugr.visualisation import progress_bar
from matplotlib import pyplot
from neodroidvision import PROJECT_APP_PATH
from neodroidvision.data.segmentation import CloudSegmentationDataset
from neodroidvision.multitask.fission.skip_hourglass import SkipHourglassFission
from neodroidvision.segmentation import (
    BCEDiceLoss,
    draw_convex_hull,
    mask_to_run_length,
)
from neodroidvision.segmentation.evaluation.iou import intersection_over_union
from pathlib import Path
from torch.utils.data import DataLoader


def post_process_minsize(mask, min_size):
    """
    Postprocessing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored"""
    num_component, component = cv2.connectedComponents(mask.astype(numpy.uint8))
    predictions, num = numpy.zeros(mask.shape), 0
    for c in range(1, num_component):
        p = component == c
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions


def threshold_mask(probability, threshold, min_size=100, psize=(350, 525)):
    """
    This is slightly different from other kernels as we draw convex hull here itself.
    Postprocessing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored"""
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    mask = draw_convex_hull(mask.astype(numpy.uint8))
    num_component, component = cv2.connectedComponents(mask.astype(numpy.uint8))
    predictions = numpy.zeros(psize, numpy.float32)
    num = 0
    for c in range(1, num_component):
        p = component == c
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


def train_model(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimiser,
    scheduler,
    save_model_path: Path,
    n_epochs=99,
):
    """

    Args:
      model:
      train_loader:
      valid_loader:
      criterion:
      optimiser:
      scheduler:
      save_model_path:
      n_epochs:

    Returns:

    """
    valid_loss_min = numpy.Inf  # track change in validation loss
    E = progress_bar(range(1, n_epochs + 1))
    for epoch in E:
        train_loss = 0.0
        valid_loss = 0.0
        dice_score = 0.0

        with TorchTrainSession(model):
            train_set = progress_bar(train_loader, postfix={"train_loss": 0.0})
            for data, target in train_set:
                data, target = (
                    data.to(global_torch_device(), dtype=torch.float),
                    target.to(global_torch_device(), dtype=torch.float),
                )
                optimiser.zero_grad()
                output, *_ = model(data)
                output = torch.sigmoid(output)
                loss = criterion(output, target)
                loss.backward()
                optimiser.step()
                train_loss += loss.item() * data.size(0)
                train_set.set_postfix(ordered_dict={"train_loss": loss.item()})

        with TorchEvalSession(model):
            with torch.no_grad():
                validation_set = progress_bar(
                    valid_loader, postfix={"valid_loss": 0.0, "dice_score": 0.0}
                )
                for data, target in validation_set:
                    data, target = (
                        data.to(global_torch_device(), dtype=torch.float),
                        target.to(global_torch_device(), dtype=torch.float),
                    )

                    output, *_ = model(
                        data
                    )  # forward pass: compute predicted outputs by passing inputs to the model
                    output = torch.sigmoid(output)

                    loss = criterion(output, target)  # calculate the batch loss

                    valid_loss += loss.item() * data.size(
                        0
                    )  # update average validation loss
                    dice_cof = intersection_over_union(
                        output.cpu().detach().numpy(), target.cpu().detach().numpy()
                    )
                    dice_score += dice_cof * data.size(0)
                    validation_set.set_postfix(
                        ordered_dict={"valid_loss": loss.item(), "dice_score": dice_cof}
                    )

        # calculate average losses
        train_loss /= len(train_loader.dataset)
        valid_loss /= len(valid_loader.dataset)
        dice_score /= len(valid_loader.dataset)

        # print training/validation statistics
        E.set_description(
            f"Epoch: {epoch}"
            f" Training Loss: {train_loss:.6f} "
            f"Validation Loss: {valid_loss:.6f} "
            f"Dice Score: {dice_score:.6f}"
        )

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print(
                f"Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}).  Saving model ..."
            )
            torch.save(model.state_dict(), str(save_model_path))
            valid_loss_min = valid_loss

        scheduler.step()

    return model


def threshold_grid_search(model, valid_loader, max_samples=2000):
    """Grid Search for best Threshold"""

    valid_masks = []
    count = 0
    tr = min(valid_loader.dataset.__len__(), max_samples)
    probabilities = numpy.zeros(
        (tr, *CloudSegmentationDataset.image_size_T), dtype=numpy.float32
    )
    for data, targets in progress_bar(valid_loader):
        data = data.to(global_torch_device(), dtype=torch.float)
        predictions, *_ = model(data)
        predictions = torch.sigmoid(predictions)
        predictions = predictions.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        for p in range(data.shape[0]):
            pred, target = predictions[p], targets[p]
            for mask_ in target:
                valid_masks.append(mask_)
            for probability in pred:
                probabilities[count, :, :] = probability
                count += 1
            if count >= tr - 1:
                break
        if count >= tr - 1:
            break

    class_params = {}

    for class_id in CloudSegmentationDataset.categories.keys():
        print(CloudSegmentationDataset.categories[class_id])
        attempts = []
        for t in range(0, 100, 5):
            t /= 100
            for ms in [0, 100, 1200, 5000, 10000, 30000]:
                masks, d = [], []
                for i in range(class_id, len(probabilities), 4):
                    probability_ = probabilities[i]
                    predict, num_predict = threshold_mask(probability_, t, ms)
                    masks.append(predict)
                for i, j in zip(masks, valid_masks[class_id::4]):
                    if (i.sum() == 0) & (j.sum() == 0):
                        d.append(1)
                    else:
                        d.append(intersection_over_union(i, j))
                attempts.append((t, ms, numpy.mean(d)))

        attempts_df = pandas.DataFrame(attempts, columns=["threshold", "size", "dice"])
        attempts_df = attempts_df.sort_values("dice", ascending=False)
        print(attempts_df.head())
        best_threshold = attempts_df["threshold"].values[0]
        best_size = attempts_df["size"].values[0]
        class_params[class_id] = (best_threshold, best_size)

    return class_params


def prepare_submission(
    model, class_params, test_loader, submission_file_path="submission.csv"
):
    """

    Args:
      model:
      class_params:
      test_loader:
      submission_file_path:
    """
    # encoded_pixels = []
    submission_i = 0
    number_of_pixels_saved = 0
    df: pandas.DataFrame = test_loader.dataset.data_frame

    with open(submission_file_path, mode="w") as f:
        f.write("Image_Label,EncodedPixels\n")
        for data, target, black_mask in progress_bar(test_loader):
            data = data.to(global_torch_device(), dtype=torch.float)
            output, *_ = model(data)
            del data
            output = torch.sigmoid(output)
            output = output.cpu().detach().numpy()
            black_mask = black_mask.cpu().detach().numpy()
            a = df["Image_Label"]
            for category in output:
                for probability in category:
                    thr, min_size = (
                        class_params[submission_i % 4][0],
                        class_params[submission_i % 4][1],
                    )
                    predict, num_predict = threshold_mask(probability, thr, min_size)
                    if num_predict == 0:
                        rle = ""
                        # encoded_pixels.append('')
                    else:
                        number_of_pixels_saved += numpy.sum(predict)
                        predict_masked = numpy.multiply(predict, black_mask)
                        number_of_pixels_saved -= numpy.sum(predict_masked)
                        rle = mask_to_run_length(predict_masked)
                        # encoded_pixels.append(rle)

                    f.write(f"{a[submission_i]},{rle}\n")
                    submission_i += 1

        # df['EncodedPixels'] = encoded_pixels
        # df.to_csv(submission_file_path, columns=['Image_Label', 'EncodedPixels'], index=False)

    print(f"Number of pixel saved {number_of_pixels_saved}")


def main():
    """description"""
    pyplot.style.use("bmh")

    base_dataset_path = Path.home() / "Data" / "Datasets" / "Clouds"
    image_path = base_dataset_path / "resized"

    save_model_path = PROJECT_APP_PATH.user_data / "cloud_seg.model"

    SEED = 87539842
    batch_size = 8
    num_workers = 0
    seed_stack(SEED)

    train_loader = DataLoader(
        CloudSegmentationDataset(
            base_dataset_path, image_path, subset=SplitEnum.training
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        CloudSegmentationDataset(
            base_dataset_path, image_path, subset=SplitEnum.validation
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        CloudSegmentationDataset(
            base_dataset_path, image_path, subset=SplitEnum.testing
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = SkipHourglassFission(
        CloudSegmentationDataset.predictor_channels,
        (CloudSegmentationDataset.response_channels,),
        encoding_depth=1,
    )
    model.to(global_torch_device())

    if save_model_path.exists():
        model.load_state_dict(torch.load(str(save_model_path)))  # load last model
        print("loading previous model")

    criterion = BCEDiceLoss(eps=1.0)
    lr = 3e-3
    optimiser = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimiser, 7, eta_min=lr / 100, last_epoch=-1
    )

    model = train_model(
        model,
        train_loader,
        valid_loader,
        criterion,
        optimiser,
        scheduler,
        save_model_path,
    )

    if save_model_path.exists():
        model.load_state_dict(torch.load(str(save_model_path)))  # load best model
    model.eval()

    class_parameters = threshold_grid_search(model, valid_loader)

    for _, (data, target) in zip(range(2), valid_loader):
        data = data.to(global_torch_device(), dtype=torch.float)
        output, *_ = model(data)
        output = torch.sigmoid(output)
        output = output[0].cpu().detach().numpy()
        image_vis = data[0].cpu().detach().numpy()
        mask = target[0].cpu().detach().numpy()

        mask = chw_to_hwc(mask)
        output = chw_to_hwc(output)
        image_vis = float_chw_to_hwc_uint(image_vis)

        pr_mask = numpy.zeros(CloudSegmentationDataset.response_shape)
        for j in range(len(CloudSegmentationDataset.categories)):
            probability_ = output[..., j]
            thr, min_size = class_parameters[j][0], class_parameters[j][1]
            pr_mask[..., j], _ = threshold_mask(probability_, thr, min_size)
        CloudSegmentationDataset.visualise_prediction(
            image_vis,
            pr_mask,
            original_image=image_vis,
            original_mask=mask,
            raw_image=image_vis,
            raw_mask=output,
        )

    prepare_submission(model, class_parameters, test_loader)


if __name__ == "__main__":
    main()
