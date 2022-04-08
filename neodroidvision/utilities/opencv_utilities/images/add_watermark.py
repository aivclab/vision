import math
from enum import Enum

import cv2
import numpy
from sorcery import assigned_names


def rotate_image(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    return cv2.warpAffine(image, cv2.getRotationMatrix2D(center, angle, scale), (w, h))


class PositionEnum(Enum):
    upper_left, lower_left, upper_right, lower_right = assigned_names()


class WatermarkImage:
    def __init__(
        self,
        logo_path,
        size=0.3,
        orientation: PositionEnum = PositionEnum.lower_right,
        margin=(5, 20, 20, 20),
        angle=15,
        rgb_weight=(0, 1, 1.5),
        input_frame_shape=None,
    ) -> None:

        logo_image = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        h, w, c = logo_image.shape
        if angle % 360 != 0:
            new_h = w * math.sin(angle / 180 * math.pi) + h * math.cos(
                angle / 180 * math.pi
            )
            pad_h = int((new_h - h) // 2)

            padding = numpy.zeros((pad_h, w, c), dtype=numpy.uint8)
            logo_image = cv2.vconcat([logo_image, padding])
            logo_image = cv2.vconcat([padding, logo_image])

            logo_image = rotate_image(logo_image, angle)
        print(logo_image.shape)
        self.logo_image = logo_image

        if self.logo_image.shape[2] < 4:
            print("No alpha channel found!")
            self.logo_image = self.__add_alpha__(self.logo_image)  # add alpha channel
        self.size = size
        self.orientation = PositionEnum(orientation)
        self.margin = margin
        self.ori_shape = self.logo_image.shape
        self.resized = False
        self.rgb_weight = rgb_weight

        self.logo_image[:, :, 2] = self.logo_image[:, :, 2] * self.rgb_weight[0]
        self.logo_image[:, :, 1] = self.logo_image[:, :, 1] * self.rgb_weight[1]
        self.logo_image[:, :, 0] = self.logo_image[:, :, 0] * self.rgb_weight[2]

        if input_frame_shape is not None:

            logo_w = input_frame_shape[1] * self.size
            ratio = logo_w / self.ori_shape[1]
            logo_h = int(ratio * self.ori_shape[0])
            logo_w = int(logo_w)

            size = (logo_w, logo_h)
            self.logo_image = cv2.resize(
                self.logo_image, size, interpolation=cv2.INTER_CUBIC
            )
            self.resized = True
            if orientation == PositionEnum.upper_left:
                self.coor_h = self.margin[1]
                self.coor_w = self.margin[0]
            elif orientation == PositionEnum.upper_right:
                self.coor_h = self.margin[1]
                self.coor_w = input_frame_shape[1] - (logo_w + self.margin[2])
            elif orientation == PositionEnum.lower_left:
                self.coor_h = input_frame_shape[0] - (logo_h + self.margin[1])
                self.coor_w = self.margin[0]
            else:
                self.coor_h = input_frame_shape[0] - (logo_h + self.margin[3])
                self.coor_w = input_frame_shape[1] - (logo_w + self.margin[2])
            self.logo_w = logo_w
            self.logo_h = logo_h
            self.mask = self.logo_image[:, :, 3]
            self.mask = cv2.bitwise_not(self.mask // 255)

    def apply_frames(self, frame):

        if not self.resized:
            shape = frame.shape
            logo_w = shape[1] * self.size
            ratio = logo_w / self.ori_shape[1]
            logo_h = int(ratio * self.ori_shape[0])
            logo_w = int(logo_w)

            size = (logo_w, logo_h)
            self.logo_image = cv2.resize(
                self.logo_image, size, interpolation=cv2.INTER_CUBIC
            )
            self.resized = True
            if self.orientation == PositionEnum.upper_left:
                self.coor_h = self.margin[1]
                self.coor_w = self.margin[0]
            elif self.orientation == PositionEnum.upper_right:
                self.coor_h = self.margin[1]
                self.coor_w = shape[1] - (logo_w + self.margin[2])
            elif self.orientation == PositionEnum.lower_left:
                self.coor_h = shape[0] - (logo_h + self.margin[1])
                self.coor_w = self.margin[0]
            else:
                self.coor_h = shape[0] - (logo_h + self.margin[3])
                self.coor_w = shape[1] - (logo_w + self.margin[2])
            self.logo_w = logo_w
            self.logo_h = logo_h
            self.mask = self.logo_image[:, :, 3]
            self.mask = cv2.bitwise_not(self.mask // 255)

        original_frame = frame[
            self.coor_h : (self.coor_h + self.logo_h),
            self.coor_w : (self.coor_w + self.logo_w),
            :,
        ]
        blending_logo = cv2.add(
            self.logo_image[:, :, 0:3], original_frame, mask=self.mask
        )
        frame[
            self.coor_h : (self.coor_h + self.logo_h),
            self.coor_w : (self.coor_w + self.logo_w),
            :,
        ] = blending_logo
        return frame

    def __add_alpha__(self, image):
        shape = image.shape
        alpha_channel = numpy.ones((shape[0], shape[1], 1), numpy.uint8) * 255
        return numpy.concatenate((image, alpha_channel), 2)
