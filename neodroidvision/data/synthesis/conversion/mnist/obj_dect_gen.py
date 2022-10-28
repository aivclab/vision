#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "{user}"
__doc__ = r"""

           Created on {date}
           """

import gzip
import pathlib
import pickle
from urllib import request

import cv2
import numpy
import tqdm

from neodroidvision import PROJECT_APP_PATH

filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"],
]
SAVE_PATH = pathlib.Path(PROJECT_APP_PATH.user_data / "Data" / "original_mnist")

__all__ = []


# __all__ = ['download_mnist','extract_mnist']


def download_mnist():
    """description"""
    SAVE_PATH.mkdir(exist_ok=True, parents=True)
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        filepath = SAVE_PATH / name[1]
        if filepath.is_file():
            continue
        print(f"Downloading {name[1]}...")
        request.urlretrieve(base_url + name[1], filepath)


def extract_mnist():
    """

    :return:
    :rtype:
    """
    save_path = SAVE_PATH / "mnist.pkl"
    if save_path.is_file():
        return
    mnist = {}
    # Load images
    for name in filename[:2]:
        path = SAVE_PATH / name[1]
        with gzip.open(path, "rb") as f:
            data = numpy.frombuffer(f.read(), numpy.uint8, offset=16)
            print(data.shape)
            mnist[name[0]] = data.reshape(-1, 28 * 28)
    # Load labels
    for name in filename[2:]:
        path = SAVE_PATH / name[1]
        with gzip.open(path, "rb") as f:
            data = numpy.frombuffer(f.read(), numpy.uint8, offset=8)
            mnist[name[0]] = data
    with open(save_path, "wb") as f:
        pickle.dump(mnist, f)


def load():
    """

    :return:
    :rtype:
    """
    download_mnist()
    extract_mnist()
    dataset_path = SAVE_PATH / "mnist.pkl"
    with open(dataset_path, "rb") as f:
        mnist = pickle.load(f)
    X_train, Y_train, X_test, Y_test = (
        mnist["training_images"],
        mnist["training_labels"],
        mnist["test_images"],
        mnist["test_labels"],
    )
    return X_train.reshape(-1, 28, 28), Y_train, X_test.reshape(-1, 28, 28), Y_test


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.
    Args:
        prediction_box (numpy.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (numpy.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]
        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE
    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = prediction_box
    if x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t:
        return 0.0

    # Compute intersection
    x1i = max(x1_t, x1_p)
    x2i = min(x2_t, x2_p)
    y1i = max(y1_t, y1_p)
    y2i = min(y2_t, y2_p)
    intersection = (x2i - x1i) * (y2i - y1i)

    # Compute union
    pred_area = (x2_p - x1_p) * (y2_p - y1_p)
    gt_area = (x2_t - x1_t) * (y2_t - y1_t)
    union = pred_area + gt_area - intersection
    iou = intersection / union
    assert 0 <= iou <= 1, f"IOU must be in [0, 1]. Got {iou}"
    return iou


def compute_iou_all(bbox, all_bboxes):
    """

    :param bbox:
    :type bbox:
    :param all_bboxes:
    :type all_bboxes:
    :return:
    :rtype:
    """
    ious = [0]
    for other_bbox in all_bboxes:
        ious.append(calculate_iou(bbox, other_bbox))
    return ious


def tight_bbox(digit, orig_bbox):
    """

    :param digit:
    :type digit:
    :param orig_bbox:
    :type orig_bbox:
    :return:
    :rtype:
    """
    xmin, ymin, xmax, ymax = orig_bbox
    # xmin
    shift = 0
    for i in range(digit.shape[1]):
        if digit[:, i].sum() != 0:
            break
        shift += 1
    xmin += shift
    # xmax
    shift = 0
    for i in range(-1, -digit.shape[1], -1):
        if digit[:, i].sum() != 0:
            break
        shift += 1
    xmax -= shift
    shift = 0
    for i in range(digit.shape[0]):
        if digit[i, :].sum() != 0:
            break
        shift += 1
    ymin += shift
    shift = 0
    for i in range(-1, -digit.shape[0], -1):
        if digit[i, :].sum() != 0:
            break
        shift += 1
    ymax -= shift
    return [xmin, ymin, xmax, ymax]


def dataset_exists(dirpath: pathlib.Path, num_images):
    """

    :param dirpath:
    :type dirpath:
    :param num_images:
    :type num_images:
    :return:
    :rtype:
    """
    if not dirpath.is_dir():
        return False
    for image_id in range(num_images):
        error_msg = f"MNIST dataset already generated in {dirpath}, \n\tbut did not find filepath:"
        error_msg2 = f"You can delete the directory by running: rm -r {dirpath.parent}"
        impath = dirpath / "images" / f"{image_id}.png"
        assert impath.is_file(), f"{error_msg} {impath} \n\t{error_msg2}"
        label_path = (dirpath / "annotations" / f"{image_id}").with_suffix(".csv")
        assert label_path.is_file(), f"{error_msg} {impath} \n\t{error_msg2}"
    return True


def generate_dataset(
    dirpath: pathlib.Path,
    num_images: int,
    max_digit_size: int,
    min_digit_size: int,
    imsize: int,
    max_digits_per_image: int,
    mnist_images: numpy.ndarray,
    mnist_labels: numpy.ndarray,
):
    """

    :param dirpath:
    :type dirpath:
    :param num_images:
    :type num_images:
    :param max_digit_size:
    :type max_digit_size:
    :param min_digit_size:
    :type min_digit_size:
    :param imsize:
    :type imsize:
    :param max_digits_per_image:
    :type max_digits_per_image:
    :param mnist_images:
    :type mnist_images:
    :param mnist_labels:
    :type mnist_labels:
    :return:
    :rtype:
    """
    if dataset_exists(dirpath, num_images):
        return
    max_image_value = 255
    assert mnist_images.dtype == numpy.uint8
    image_dir = dirpath / "images"
    label_dir = dirpath / "annotations"
    image_dir.mkdir(exist_ok=True, parents=True)
    label_dir.mkdir(exist_ok=True, parents=True)
    for image_id in tqdm.trange(
        num_images, description=f"Generating dataset, saving to: {dirpath}"
    ):
        im = numpy.zeros((imsize, imsize), dtype=numpy.float32)
        labels = []
        bboxes = []
        num_images = numpy.random.randint(0, max_digits_per_image)
        for _ in range(num_images + 1):
            while True:
                width = numpy.random.randint(min_digit_size, max_digit_size)
                x0 = numpy.random.randint(0, imsize - width)
                y0 = numpy.random.randint(0, imsize - width)
                ious = compute_iou_all([x0, y0, x0 + width, y0 + width], bboxes)
                if max(ious) < 0.25:
                    break
            digit_idx = numpy.random.randint(0, len(mnist_images))
            digit = mnist_images[digit_idx].astype(numpy.float32)
            digit = cv2.resize(digit, (width, width))
            label = mnist_labels[digit_idx]
            labels.append(label)
            assert (
                im[y0 : y0 + width, x0 : x0 + width].shape == digit.shape
            ), f"imshape: {im[y0:y0 + width, x0:x0 + width].shape}, digit shape: {digit.shape}"
            bbox = tight_bbox(digit, [x0, y0, x0 + width, y0 + width])
            bboxes.append(bbox)

            im[y0 : y0 + width, x0 : x0 + width] += digit
            im[im > max_image_value] = max_image_value
        image_target_path = (image_dir / f"{image_id}").with_suffix(".png")
        label_target_path = (label_dir / f"{image_id}").with_suffix(".csv")
        im = im.astype(numpy.uint8)
        cv2.imwrite(str(image_target_path), im)
        with open(label_target_path, "w") as fp:
            fp.write("label,xmin,ymin,xmax,ymax\n")
            for l, bbox in zip(labels, bboxes):
                bbox = [str(_) for _ in bbox]
                to_write = f"{l}," + ",".join(bbox) + "\n"
                fp.write(to_write)


if __name__ == "__main__":

    def main(
        base_path=PROJECT_APP_PATH.user_data / "Data" / "mnist_detection",
        imsize=300,
        max_digit_size=100,
        min_digit_size=15,
        num_train_images=10000,
        num_test_images=1000,
        max_digits_per_image=20,
    ):
        """

        :param base_path:
        :type base_path:
        :param imsize:
        :type imsize:
        :param max_digit_size:
        :type max_digit_size:
        :param min_digit_size:
        :type min_digit_size:
        :param num_train_images:
        :type num_train_images:
        :param num_test_images:
        :type num_test_images:
        :param max_digits_per_image:
        :type max_digits_per_image:
        """
        X_train, Y_train, X_test, Y_test = load()
        for dataset, (X, Y) in zip(
            ["train", "test"], [[X_train, Y_train], [X_test, Y_test]]
        ):
            num_images = num_train_images if dataset == "train" else num_test_images
            generate_dataset(
                pathlib.Path(base_path, dataset),
                num_images,
                max_digit_size,
                min_digit_size,
                imsize,
                max_digits_per_image,
                X,
                Y,
            )

        main()
