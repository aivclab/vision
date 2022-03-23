from collections import defaultdict
from pathlib import Path

import cv2
import numpy
from draugr.opencv_utilities import (
    show_image,
    to_gray,
    noise_filter,
    NoiseFilterMethodEnum,
    TermCriteriaFlag,
    KmeansEnum,
    ColorConversionEnum,
    is_singular_channel,
)

__all__ = ["draw_parameter_lines", "draw_markers", "find_intersections"]


def draw_parameter_lines(img, lines, color=(0, 0, 255)):
    """
    Draw lines on an image
    """
    if is_singular_channel(img):
        img = cv2.cvtColor(img, ColorConversionEnum.gray2bgr.value)

    for line in lines:
        for rho, theta in line:
            a = numpy.cos(theta)
            b = numpy.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img, (x1, y1), (x2, y2), color, 1)
    return img


def draw_markers(img, centroids, length=5):
    if is_singular_channel(img):
        img = cv2.cvtColor(img, ColorConversionEnum.gray2bgr.value)

    for ct in centroids:
        ct = numpy.array(ct, dtype=numpy.int32)
        cv2.line(
            img,
            (ct[0], ct[1] - length),
            (ct[0], ct[1] + length),
            (255, 0, 255),
            1,
        )  # vertical line
        cv2.line(
            img,
            (ct[0] - length, ct[1]),
            (ct[0] + length, ct[1]),
            (255, 0, 255),
            1,
        )

    return img


def find_centroids(pts, k=4, **kwargs):
    assert len(pts) >= k
    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = TermCriteriaFlag.eps.value + TermCriteriaFlag.max_iter.value
    criteria = kwargs.get("criteria", (default_criteria_type, 10, 1.0))

    flags = kwargs.get("flags", KmeansEnum.random_centers.value)
    attempts = kwargs.get("attempts", 10)

    return cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]


def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """
    Group lines by their angle using k-means clustering.

    Code from here:
    https://stackoverflow.com/a/46572063/1755401
    """

    # Get angles in [0, pi] radians
    angles = numpy.array([line[0][1] for line in lines])

    # Multiply the angles by two and find coordinates of that angle on the Unit Circle
    pts = numpy.array(
        [[numpy.cos(2 * angle), numpy.sin(2 * angle)] for angle in angles],
        dtype=numpy.float32,
    )

    labels, centers = find_centroids(pts, k)
    labels = labels.reshape(-1)  # Transpose to row vector

    # Segment lines based on their label of 0 or 1
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)

    segmented = list(segmented.values())
    # print(f"Segmented lines into two groups: {len(segmented[0]):d}, {len(segmented[1]):d}")

    return segmented


def intersection(line1, line2):
    """
    Find the intersection of two lines
    specified in Hesse normal form.

    Returns the closest integer pixel locations.

    See here:
    https://stackoverflow.com/a/383527/5087436
    """

    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = numpy.array(
        [
            [numpy.cos(theta1), numpy.sin(theta1)],
            [numpy.cos(theta2), numpy.sin(theta2)],
        ]
    )
    b = numpy.array([[rho1], [rho2]])
    x0, y0 = numpy.linalg.solve(A, b)

    return [[int(numpy.round(x0)), int(numpy.round(y0))]]


def segmented_intersections(lines):
    """
    Find the intersection between groups of lines.
    """
    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i + 1 :]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))

    return intersections


def find_intersections(lines, k_angles=2, k_centers=4):
    """
    Find the intersection points of lines.
    """

    segmented = segment_by_angle_kmeans(
        lines, k=k_angles
    )  # Cluster line angles into 2 groups (vertical and horizontal)
    intersections = segmented_intersections(
        segmented
    )  # Find the intersections of each vertical line with each horizontal line

    points = []
    for point in intersections:
        pt = (point[0][0], point[0][1])
        points.append(numpy.array(pt))

    labels, centers = find_centroids(
        numpy.array(points, dtype=numpy.float32), k=k_centers
    )

    return centers


if __name__ == "__main__":
    # file = "360_F_108702068_Z9VGab1DfiyPzq2v5Xgm2wRljttzRGgq.jpg"
    # file = "istockphoto-529081402-170667a.jpg"
    # file = "white-paper-table-13912022.jpg"
    # file = "2.jpg"
    file = "3.jpg"
    # file = "NdNLO.jpg"
    image = cv2.imread(str(Path.home() / "Pictures" / file))
    gray = to_gray(image)

    blur = noise_filter(gray, method=NoiseFilterMethodEnum.median_blur, ksize=5)

    adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh_type = cv2.THRESH_BINARY_INV
    bin_img = cv2.adaptiveThreshold(blur, 255, adapt_type, thresh_type, 11, 2)
    if False:
        show_image(bin_img, wait=True)

    lines = cv2.HoughLines(
        bin_img, rho=2, theta=numpy.pi / 180, threshold=350
    )  # Detect lines
    """
    if False:
        img_with_all_lines = numpy.copy(img)
        draw_lines(img_with_all_lines, lines)
        show_image(img_with_all_lines, wait=True)
    """
    centroids = find_intersections(lines)
    show_image(draw_markers(image.copy(), centroids), wait=True)
