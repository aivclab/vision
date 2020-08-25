#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import copy
import os
import time

import torch
import torchvision
from matplotlib import pyplot
from neodroidvision import PROJECT_APP_PATH
from neodroidvision.classification import squeezenet_retrain
from neodroidvision.data.classification.deprec.s_mnist import MNISTDataset
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from draugr import horizontal_imshow, recycle
from draugr.torch_utilities import (
    Split,
    TensorBoardPytorchWriter,
    TorchEvalSession,
    TorchTrainSession,
    ensure_directory_exist,
    global_torch_device,
    to_tensor,
    torch_clean_up,
)

__author__ = "Christian Heider Nielsen"
__all__ = []

seed = 34874312
batch_size = 16
tqdm.monitor_interval = 0
learning_rate = 3e-5
momentum = 0.9
wd = 3e-8
test_batch_size = batch_size
early_stop = 3e-6
num_updates = 6000
lr_cycles = 1
flatt_size = 224 * 224 * 3

normalise = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
)


def predictor_response_train_model(
    model,
    *,
    train_iterator,
    criterion,
    optimizer,
    scheduler,
    writer,
    interrupted_path,
    val_data_iterator=None,
    num_updates: int = 250000,
    device=global_torch_device(),
    early_stop=None,
):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = 1e10
    since = time.time()

    try:
        sess = tqdm(range(num_updates), leave=False, disable=False)
        val_loss = 0
        update_loss = 0
        val_acc = 0
        last_val = None
        last_out = None
        with torch.autograd.detect_anomaly():
            for update_i in sess:
                for phase in [Split.Training, Split.Validation]:
                    if phase == Split.Training:
                        with TorchTrainSession(model):

                            input, true_label = next(train_iterator)

                            rgb_imgs = to_tensor(
                                input, dtype=torch.float, device=device
                            ).repeat(1, 3, 1, 1)
                            true_label = to_tensor(
                                true_label, dtype=torch.long, device=device
                            )
                            optimizer.zero_grad()

                            pred = model(rgb_imgs)
                            loss = criterion(pred, true_label)
                            loss.backward()
                            optimizer.step()

                            update_loss = loss.data.cpu().numpy()
                            writer.scalar(f"loss/train", update_loss, update_i)

                            if scheduler:
                                scheduler.step()
                    elif val_data_iterator:
                        with TorchEvalSession(model):
                            test_rgb_imgs, test_true_label = next(val_data_iterator)

                            test_rgb_imgs = to_tensor(
                                test_rgb_imgs, dtype=torch.float, device=device
                            ).repeat(1, 3, 1, 1)
                            test_true_label = to_tensor(
                                test_true_label, dtype=torch.long, device=device
                            )

                            with torch.no_grad():
                                val_pred = model(test_rgb_imgs)
                                val_loss = criterion(val_pred, test_true_label)

                            _, cat = torch.max(val_pred, -1)
                            val_acc = torch.sum(cat == test_true_label) / float(
                                cat.size(0)
                            )
                            writer.scalar(f"loss/acc", val_acc, update_i)
                            writer.scalar(f"loss/val", val_loss, update_i)

                            if val_loss < best_val_loss:
                                best_val_loss = val_loss

                                best_model_wts = copy.deepcopy(model.state_dict())
                                sess.write(
                                    f"New best validation model at update {update_i} with best_val_loss {best_val_loss}"
                                )
                                torch.save(model.state_dict(), interrupted_path)

                        if early_stop is not None and val_pred < early_stop:
                            break
                sess.set_description_str(
                    f"Update {update_i} - {phase} "
                    f"update_loss:{update_loss:2f} "
                    f"val_loss:{val_loss}"
                    f"val_acc:{val_acc}"
                )

    except KeyboardInterrupt:
        print("Interrupt")
    finally:
        pass

    model.load_state_dict(best_model_wts)  # load best model weights

    time_elapsed = time.time() - since
    print(f"{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val loss: {best_val_loss:3f}")

    return model


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--inference", "-i", action="store_true")
    args.add_argument("--continue_training", "-c", action="store_true")
    args.add_argument("--real_data", "-r", action="store_true")
    args.add_argument("--no_cuda", "-k", action="store_false")
    args.add_argument("--export", "-e", action="store_true")
    options = args.parse_args()

    train_model = True
    timeas = str(time.time())
    this_model_path = PROJECT_APP_PATH.user_data / timeas
    this_log = PROJECT_APP_PATH.user_log / timeas
    ensure_directory_exist(this_model_path)
    ensure_directory_exist(this_log)

    best_model_name = "best_validation_model.model"
    interrupted_path = str(this_model_path / best_model_name)

    torch.manual_seed(seed)

    if not options.no_cuda:
        global_torch_device("cpu")

    dataset = MNISTDataset(PROJECT_APP_PATH.user_cache / "mnist", split=Split.Training)
    train_iter = iter(
        recycle(
            DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        )
    )

    val_iter = iter(
        recycle(
            DataLoader(
                MNISTDataset(
                    PROJECT_APP_PATH.user_cache / "mnist", split=Split.Validation
                ),
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
            )
        )
    )

    model, params_to_update = squeezenet_retrain(len(dataset.categories))
    print(params_to_update)
    model = model.to(global_torch_device())

    if options.continue_training:
        _list_of_files = PROJECT_APP_PATH.user_data.rglob("*.model")
        latest_model_path = str(max(_list_of_files, key=os.path.getctime))
        print(f"loading previous model: {latest_model_path}")
        if latest_model_path is not None:
            model.load_state_dict(torch.load(latest_model_path))

    criterion = torch.nn.CrossEntropyLoss().to(global_torch_device())

    optimizer_ft = optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=wd
    )
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1
    )

    writer = TensorBoardPytorchWriter(this_log)

    if train_model:
        model = predictor_response_train_model(
            model,
            train_iterator=train_iter,
            criterion=criterion,
            optimizer=optimizer_ft,
            scheduler=exp_lr_scheduler,
            writer=writer,
            interrupted_path=interrupted_path,
            val_data_iterator=val_iter,
            num_updates=num_updates,
        )

    inputs, true_label = next(train_iter)
    inputs = to_tensor(inputs, dtype=torch.float, device=global_torch_device()).repeat(
        1, 3, 1, 1
    )
    true_label = to_tensor(true_label, dtype=torch.long, device=global_torch_device())

    pred = model(inputs)
    predicted = torch.argmax(pred, -1)
    true_label = to_tensor(true_label, dtype=torch.long)
    print(predicted, true_label)
    horizontal_imshow(
        inputs, [f"p:{int(p)},t:{int(t)}" for p, t in zip(predicted, true_label)]
    )
    pyplot.show()

    writer.close()
    torch_clean_up()

    # model.eval()
    # example = torch.rand(1, 3, 256, 256)
    # traced_script_module = torch.jit.trace(model.to("cpu"), example)
    # traced_script_module.save("resnet18_v.model")


if __name__ == "__main__":
    main()
