from enum import Enum

import cv2
from sorcery import assigned_names


__all__ = ["ApproximateShapeEnum", "approximate_shape"]


class ApproximateShapeEnum(Enum):
    triangle, square, rectangle, pentagon, circle = assigned_names()


def approximate_shape(contour, threshold=0.04) -> ApproximateShapeEnum:
    approximation = cv2.approxPolyDP(
        contour, threshold * cv2.arcLength(contour, True), True
    )

    if len(approximation) == 3:  # if the shape is a triangle, it will have 3 vertices
        return ApproximateShapeEnum.triangle

    elif len(approximation) == 4:  # if the shape has 4 vertices, it is a rectangle
        (x, y, w, h) = cv2.boundingRect(
            approximation
        )  # compute the bounding box of the contour and use the
        ar = w / float(h)  # bounding box to compute the aspect ratio
        if ar >= 0.95 and ar <= 1.05:
            return ApproximateShapeEnum.square
        return ApproximateShapeEnum.rectangle

    elif len(approximation) == 5:  # if the shape is a pentagon, it will have 5 vertices
        return ApproximateShapeEnum.pentagon

    return ApproximateShapeEnum.circle  # otherwise, we assume the shape is a circle


if __name__ == "__main__":

    def isajd():
        from pathlib import Path
        import numpy

        file = "360_F_108702068_Z9VGab1DfiyPzq2v5Xgm2wRljttzRGgq.jpg"
        # file = "NdNLO.jpg"

        image = cv2.imread(str(Path.home() / "Pictures" / file))
        # iasjdisajhd(image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        invGamma = 1.0 / 0.3

        table = numpy.array(
            [((i / 255.0) ** invGamma) * 255 for i in numpy.arange(0, 256)]
        ).astype("uint8")

        gray = cv2.LUT(gray, table)  # apply gamma correction using the lookup table

        ret, thresh1 = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

        contours, hier = cv2.findContours(
            thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        for c in contours:
            print(approximate_shape(c))

    isajd()
