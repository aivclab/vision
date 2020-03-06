#!/usr/bin/env python
# coding: utf-8
import math
import time

import numpy
import torch
from PIL import Image, ImageDraw
from matplotlib import animation, pyplot
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor

from neodroid.environments.unity_environment import UnityEnvironment
from neodroid.environments.unity_environment.deprecated.batched_unity_environments import (
    VectorWrapper,
)
from neodroid.utilities import extract_all_as_camera
from warg.named_ordered_dictionary import NOD

cudnn = torch.backends.cudnn
cudnn.benchmark = True
cudnn.enabled = True

rpn_n = 4
"""
from torchvision.models.detection import fasterrcnn_resnet50_fpn
model = fasterrcnn_resnet50_fpn(pretrained=True,
                                  min_size=128,
                                  rpn_pre_nms_top_n_test=rpn_n,
                                  rpn_post_nms_top_n_test=max(1, rpn_n // 2),
                                  box_score_thresh=0.5,
                                  box_detections_per_img=5)
"""
model = keypointrcnn_resnet50_fpn(
    pretrained=True,
    min_size=128,
    rpn_pre_nms_top_n_test=rpn_n,
    rpn_post_nms_top_n_test=max(1, rpn_n // 2),
    box_score_thresh=0.5,
    box_detections_per_img=3,
)
model.eval()


# model.cuda() # Construct the network and move to GPU


def get_preds(img_t: torch.Tensor, threshold=0.7):
    """
    Make `img` a tensor, transfer to GPU and run inference.
    Returns bounding boxes and keypoints for each person.
"""
    with torch.no_grad():
        if next(model.parameters()).is_cuda:
            img_t = img_t.pin_memory().cuda(non_blocking=True)
        pred = model(img_t)
        pred = pred[0]

    boxes = pred["boxes"]
    kpts = pred["keypoints"]
    box_scores = pred["scores"]
    kpt_scores = pred["keypoints_scores"]
    idxs = [i for (i, s) in enumerate(box_scores) if s > threshold]
    res = [(boxes[i].cpu().numpy(), kpts[i].cpu().numpy()) for i in idxs]
    return res


def show_preds(img, pred):
    """
    Draw bounding boxes and keypoints.
"""
    draw = ImageDraw.Draw(img)
    drawdot = lambda x, y, r=3, fill="red": draw.ellipse(
        (x - r, y - r, x + r, y + r), fill=fill
    )
    for (box, kpts) in pred:
        for kpt in kpts:
            if kpt[2] == 1:
                drawdot(kpt[0], kpt[1])
        draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline="red", width=2)
    return img


def to_dict_detections(preds):
    """
    Return predictions
"""
    names = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]
    detections = [keypoints for (_, keypoints) in preds]
    res = []
    for kpts in detections:
        d = {n: k.round().astype(int).tolist() for (n, k) in zip(names, kpts)}
        res.append(d)
    return res


def grab_video_frame(cap):
    ret, frame = cap.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


frame_i = 0
time_s = time.time()

image_axs = NOD()

env = VectorWrapper(UnityEnvironment(connect_to_running=True))
fig = pyplot.figure()
print_obs = False


def update_figures(i):
    global time_s, frame_i, image_axs

    sample = env.action_space.sample()
    obs, signal, terminated, info = env.react(sample).to_gym_like_output()
    if print_obs:
        print(i)
        for obs in info.sensors.values():
            print(obs)

    new_images = extract_all_as_camera(info)

    # new_images['RGB'] = new_images['RGB'] ** 0.454545

    # print( numpy.max(new_images['RGB']))

    time_now = time.time()
    if time_s:
        fps = 1 / (time_now - time_s)
    else:
        fps = 0

    time_s = time_now

    fig.suptitle(
        f"Update: {i}, "
        f"Frame: {frame_i}, "
        f"FPS: {fps}, "
        f"Signal: {signal}, "
        f"Terminated: {bool(terminated)}"
    )

    for k, v in new_images.items():
        v = v * 255.0
        v = numpy.ascontiguousarray(v.astype(numpy.uint8))
        t = Image.fromarray(v, mode="RGBA").convert("RGB")
        img_t = to_tensor(t).unsqueeze(0)
        preds = get_preds(img_t, 0.70)
        v_out = show_preds(t, preds)

        image_axs[k].set_data(v_out)

    if terminated:
        env.reset()
        frame_i = 0
    else:
        frame_i += 1


def main():
    global image_axs

    env.reset()
    acs = env.action_space.sample()
    obs, rew, term, info = env.react(acs).to_gym_like_output()
    if print_obs:
        print(0)
        for obs in info.sensors.values():
            print(obs)

    new_images = extract_all_as_camera(info)

    xs = math.floor(len(new_images) // 2) + 1
    ys = 1

    axes = fig.subplots(ys, xs, sharex="all", sharey="all")

    if ys > 1:
        a = axes.flatten()

        for ax, (k, v) in zip(a, new_images.items()):
            if k:
                ax.set_title(k)
                image_axs[k] = ax.imshow(v)
    else:
        ax = axes
        for k, v in new_images.items():
            if k:
                ax.set_title(k)
                image_axs[k] = ax.imshow(v)

    _ = animation.FuncAnimation(fig, update_figures)
    pyplot.show()


if __name__ == "__main__":
    main()
