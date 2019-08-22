#!/usr/bin/env python
# coding: utf-8
import torch
from fastai.core import defaults
from fastai.vision import ImageImageList, imagenet_stats, models, rand_resize_crop, unet_learner

from apppath import AppPath

defaults.device = torch.device('cpu')

path = AppPath("RayTraceDenoise").user_data / 'app' / 'rend_data' / 'input'

# exts = ["_real.JPG", "_mean.JPG"]
exts = ["_lowres.png", ".png"]


def get_y_fn(x):
  y = (x.parent.parent / "gt" / str(x.name).replace(exts[0], exts[1]))
  # y = (x.parent / x.stem).with_suffix(".npz")
  return y


# Construct data generator
tfms = rand_resize_crop(128, max_scale=1.0, ratios=(1, 1))
# tfms = zoom_crop(scale = 0.3, do_rand=True, p=0.0)

# _tfms = get_transforms(do_flip=True, flip_vert=True, max_zoom=0, p_affine=0, p_lighting=0)
data = (ImageImageList.from_folder(path)
        .split_by_rand_pct(0.05)
        .label_from_func(get_y_fn)
        .transform((tfms, tfms), tfm_y=True)
        .databunch(bs=2)
        .normalize(imagenet_stats))

# x = ImageImageList.from_folder(path).split_by_rand_pct(0.05).label_from_func(get_y_fn)
# x.transform((tfms,tfms), tfm_y = True)


learn = unet_learner(data,
                     models.resnet18,
                     wd=1e-2,
                     last_cross=True,
                     bottle=True,
                     self_attention=True)  # ,y_range=(0,1))
# learn.loss_func = feat_loss #MSELossFlat(axis=1)
learn.load(str(path / "models/second_model"))

# print(learn.model)
# print(torch.__version__)
import torch.onnx as torch_onnx

input_shape = (3, 512, 512)
model_onnx_path = "torch_model.onnx"

# Export the model to an ONNX file
dummy_input = torch.randn(2, *input_shape)
output = torch_onnx.export(learn.model,
                           dummy_input,

                           model_onnx_path,
                           opset_version=10,
                           export_params=True,
                           do_constant_folding=True,
                           verbose=True)
print("Export of torch_model.onnx complete!")
