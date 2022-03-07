from pathlib import Path

import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from skimage import io
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage import color

file = "3.jpg"
# file = "NdNLO.jpg"
# image = cv2.imread(str(Path.home() / "Pictures" / file))

# Constructing test image
image = color.rgb2gray(io.imread(str(Path.home() / "Pictures" / file)))

# Classic straight-line Hough transform
# Set a precision of 0.05 degree.
tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 3600)

h, theta, d = hough_line(image, theta=tested_angles)
hpeaks = hough_line_peaks(h, theta, d, threshold=0.2 * h.max())

fig, ax = plt.subplots()
ax.imshow(image, cmap=cm.gray)

for _, angle, dist in zip(*hpeaks):
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    ax.axline((x0, y0), slope=np.tan(angle + np.pi / 2))

plt.show()
