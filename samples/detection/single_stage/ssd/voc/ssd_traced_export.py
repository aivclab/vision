#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/03/2020
           """

import argparse
import cv2
import torch
from draugr.numpy_utilities import SplitEnum
from draugr.opencv_utilities import frame_generator
from draugr.torch_utilities import global_torch_device
from neodroidvision import PROJECT_APP_PATH
from neodroidvision.detection.single_stage.ssd.architecture import SingleShotDetection
from neodroidvision.detection.single_stage.ssd.bounding_boxes.ssd_transforms import (
    SSDTransform,
)
from neodroidvision.utilities.torch_utilities.persistence.check_pointer import (
    CheckPointer,
)
from pathlib import Path
from torch import onnx, quantization
from warg import NOD, ensure_existence, sprint


@torch.no_grad()
def export_detection_model(
    cfg: NOD,
    model_checkpoint: Path,
    model_export_path: Path = Path("torch_model"),
    verbose: bool = True,
    onnx_export: bool = False,
    strict_jit: bool = False,
) -> None:
    """

    :param onnx_export:
    :type onnx_export:
    :param strict_jit:
    :type strict_jit:
    :param verbose:
    :type verbose:
    :param cfg:
    :type cfg:
    :param model_checkpoint:
    :type model_checkpoint:
    :param model_export_path:
    :type model_export_path:
    :return:
    :rtype:"""
    model = SingleShotDetection(cfg)

    checkpointer = CheckPointer(
        model, save_dir=ensure_existence(PROJECT_APP_PATH.user_data / "results")
    )
    checkpointer.load(model_checkpoint, use_latest=model_checkpoint is None)
    print(
        f"Loaded weights from {model_checkpoint if model_checkpoint else checkpointer.get_checkpoint_file()}"
    )

    model.post_init()
    model.to(global_torch_device())

    transforms = SSDTransform(
        cfg.input.image_size, cfg.input.pixel_mean, split=SplitEnum.testing
    )
    model.eval()  # Important!

    fuse_quantize_model = False
    if fuse_quantize_model:
        modules_to_fuse = [
            ["conv", "bn", "relu"]
        ]  # Names of modules to fuse, maybe supply directly for architecture class/declaration
        model = torch.quantization.fuse_modules(
            model, modules_to_fuse=modules_to_fuse, inplace=False
        )

    pre_quantize_model = False
    if pre_quantize_model:  # Accuracy may drop!
        if True:
            model = quantization.quantize_dynamic(model, dtype=torch.qint8)
        else:
            pass
            # model = quantization.quantize(model)

    frame_g = frame_generator(cv2.VideoCapture(0))
    for image in progress_bar(frame_g):
        example_input = (transforms(image)[0].unsqueeze(0).to(global_torch_device()),)
        try:
            if onnx_export:
                exp_path = model_export_path.with_suffix(".onnx")
                output = onnx.export(
                    model,
                    example_input,
                    str(exp_path),
                    verbose=verbose,
                    # export_params=True,  # store the trained parameter weights inside the model file
                    # opset_version=10,  # the onnx version to export the model to
                    # do_constant_folding=True,  # wether to execute constant folding for optimization
                    # input_names=["input"],  # the model's input names
                    # output_names=["output"],  # the model's output names
                    # dynamic_axes={
                    #  "input": {0: "batch_size"},  # variable lenght axes
                    #  "output": {0: "batch_size"},
                    #  }
                )
                sprint(f"Successfully exported ONNX model at {exp_path}", color="blue")
            else:
                raise Exception("Just trace instead, ignore exception")
        except Exception as e:
            sprint(f"Torch ONNX export does not work, {e}", color="red")
            try:
                traced_script_module = torch.jit.trace(
                    model,
                    example_input,
                    # strict=strict_jit,
                    check_inputs=(
                        transforms(next(frame_g))[0]
                        .unsqueeze(0)
                        .to(global_torch_device()),
                        transforms(next(frame_g))[0]
                        .unsqueeze(0)
                        .to(global_torch_device()),
                    ),
                )
                exp_path = model_export_path.with_suffix(".traced")
                traced_script_module.save(str(exp_path))
                print(
                    f"Traced Ops used {torch.jit.export_opnames(traced_script_module)}"
                )
                sprint(
                    f"Successfully exported JIT Traced model at {exp_path}",
                    color="green",
                )
            except Exception as e_i:
                sprint(f"Torch JIT Trace export does not work!, {e_i}", color="red")

        break

    """
post_quantize_model = False
if post_quantize_model: # Accuracy may drop!
traced_model = model
if True:
q_model=quantization.prepare_script(traced_model)
... qmodel.forward(...) .. training
q_model=quantization.convert_script(traced_model)
q_model.save('model.qtraced')
"""


def main():
    """description"""
    from configs.mobilenet_v2_ssd320_voc0712 import base_cfg

    # from configs.efficient_net_b3_ssd300_voc0712 import base_cfg
    # from configs.vgg_ssd300_coco_trainval35k import base_cfg
    # from .configs.vgg_ssd512_coco_trainval35k import base_cfg

    global_torch_device(override=global_torch_device("cpu"))

    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=PROJECT_APP_PATH.user_data
        / "ssd"
        / "models"
        / "mobilenet_v2_ssd320_voc0712.pth"
        # "mobilenet_v2_ssd320_voc0712.pth"
        # "vgg_ssd300_coco_trainval35k.pth"
        # "vgg_ssd512_coco_trainval35k.pth"
        ,
        help="Trained " "weights.",
    )
    args = parser.parse_args()

    export_detection_model(cfg=base_cfg, model_checkpoint=Path(args.ckpt))


if __name__ == "__main__":
    main()
