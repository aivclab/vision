#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import copy
import os
import time
from pathlib import Path

from draugr import global_torch_device, hwc_to_chw
from neodroid.wrappers.observation_wrapper import CameraObservationWrapper
from neodroidvision.multitask import SkipHourglassFission
from neodroidvision.segmentation.segmentation_utilities.masks import plot_utilities

__author__ = "Christian Heider Nielsen"

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from matplotlib import pyplot
from draugr.writers import TensorBoardPytorchWriter, ImageWriter
from samples.segmentation.dmr.dmr_data import (
    calculate_loss,
    neodroid_camera_data_iterator,
)


def get_metric_str(metrics, writer: ImageWriter, update_i):
    outputs = []
    for k, v in metrics:
        a = v.data.cpu().numpy()
        writer.scalar(f"loss/{k}", a, update_i)
        outputs.append(f"{k}:{a:2f}")

    return f'{", ".join(outputs)}'


def train_model(
    model,
    data_iterator,
    optimizer,
    scheduler,
    writer: ImageWriter,
    interrupted_path,
    num_updates=25000,
):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    since = time.time()

    try:
        sess = tqdm(range(num_updates), leave=False)
        for update_i in sess:
            for phase in ["train", "val"]:
                if phase == "train":
                    scheduler.step()
                    for param_group in optimizer.param_groups:
                        writer.scalar("lr", param_group["lr"], update_i)

                    model.train()
                else:
                    model.eval()

                rgb_imgs, (seg_target, depth_target, normals_target) = next(
                    data_iterator
                )

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    seg_pred, recon_pred, depth_pred, normals_pred = model(rgb_imgs)
                    ret = calculate_loss(
                        (seg_pred, seg_target),
                        (recon_pred, rgb_imgs),
                        (depth_pred, depth_target),
                        (normals_pred, normals_target),
                    )

                    if phase == "train":
                        ret.loss.backward()
                        optimizer.step()

                update_loss = ret.loss.data.cpu().numpy()
                writer.scalar(f"loss/accum", update_loss, update_i)

                if phase == "val" and update_loss < best_loss:
                    best_loss = update_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    writer.image(f"rgb_imgs", rgb_imgs, update_i)
                    writer.image(f"recon_pred", recon_pred, update_i)
                    writer.image(f"seg_target", seg_target, update_i)
                    writer.image(f"seg_pred", seg_pred, update_i)
                    writer.image(
                        f"depth_target", depth_target, update_i, dataformats="NCHW"
                    )
                    writer.image(
                        f"depth_pred", depth_pred, update_i, dataformats="NCHW"
                    )
                    writer.image(f"normals_pred", normals_pred, update_i)
                    writer.image(f"normals_target", normals_target, update_i)
                    sess.write(f"New best model at update {update_i}")

            _ = get_metric_str(ret.terms, writer, update_i)
            sess.set_description_str(
                f"Update {update_i} - {phase} accum_loss:{update_loss:2f}"
            )

            if update_loss < 0.1:
                break
    except KeyboardInterrupt:
        print("Interrupt")
    finally:
        model.load_state_dict(best_model_wts)  # load best model weights
        torch.save(model.state_dict(), interrupted_path)

    time_elapsed = time.time() - since
    print(f"{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val loss: {best_loss:3f}")

    return model


def test_model(model, data_iterator, load_path=None):
    if load_path is not None:
        model.load_state_dict(torch.load(load_path))

    model.eval()

    inputs, (labels, _, _) = next(data_iterator)

    pred, recon, _, _ = model(inputs)
    pred = pred.data.cpu().numpy()
    recon = recon.data.cpu().numpy()
    l = labels.cpu().numpy()
    inputs = inputs.cpu().numpy()

    input_images_rgb = [hwc_to_chw(x) for x in inputs]
    target_masks_rgb = [plot_utilities.masks_to_color_img(hwc_to_chw(x)) for x in l]
    pred_rgb = [plot_utilities.masks_to_color_img(hwc_to_chw(x)) for x in pred]
    pred_recon = [hwc_to_chw(x) for x in recon]

    plot_utilities.plot_side_by_side(
        [input_images_rgb, target_masks_rgb, pred_rgb, pred_recon]
    )
    pyplot.show()


def main():
    args = argparse.ArgumentParser()
    args.add_argument("-i", action="store_false")
    options = args.parse_args()

    seed = 42
    batch_size = 8  # 12
    depth = 4  # 5
    segmentation_channels = 3
    tqdm.monitor_interval = 0
    learning_rate = 3e-3
    lr_sch_step_size = int(1000 // batch_size) + 4
    lr_sch_gamma = 0.1
    model_start_channels = 16

    home_path = Path.home() / "Models" / "Vision"
    base_path = home_path / str(time.time())
    best_model_path = "INTERRUPTED_BEST.pth"
    interrupted_path = str(base_path / best_model_path)

    writer = TensorBoardPytorchWriter(str(base_path))
    env = CameraObservationWrapper()

    torch.manual_seed(seed)
    env.seed(seed)

    device = global_torch_device()

    aeu_model = SkipHourglassFission(
        segmentation_channels,
        (segmentation_channels,),
        encoding_depth=depth,
        start_channels=model_start_channels,
    )
    aeu_model = aeu_model.to(global_torch_device())

    optimizer_ft = optim.Adam(aeu_model.parameters(), lr=learning_rate)

    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=lr_sch_step_size, gamma=lr_sch_gamma
    )

    data_iter = iter(neodroid_camera_data_iterator(env, device, batch_size))

    if options.i:
        trained_aeu_model = train_model(
            aeu_model,
            data_iter,
            optimizer_ft,
            exp_lr_scheduler,
            writer,
            interrupted_path,
        )
        test_model(trained_aeu_model, data_iter)
    else:
        _list_of_files = home_path.glob("*")
        lastest_model_path = (
            str(max(_list_of_files, key=os.path.getctime)) + f"/{best_model_path}"
        )
        print("loading previous model: " + lastest_model_path)
        test_model(aeu_model, data_iter, load_path=lastest_model_path)

    torch.cuda.empty_cache()
    env.close()
    writer.close()


if __name__ == "__main__":
    main()
