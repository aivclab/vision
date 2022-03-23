import cv2
import numpy
from draugr.opencv_utilities import show_image, to_gray

__all__ = ["skeletonise", "top_skeleton", "thin"]


def top_skeleton(img):
    size = numpy.size(img)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    skel = numpy.zeros(img.shape, numpy.uint8)

    while not done:
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, cv2.subtract(img, cv2.dilate(eroded, element)))
        img = eroded.copy()

        if size - cv2.countNonZero(img) == size:
            done = True

    return skel


def skeletonise(image):
    return cv2.ximgproc.thinning(image)


thin = skeletonise

if __name__ == "__main__":

    def uahsd():
        # file = "360_F_108702068_Z9VGab1DfiyPzq2v5Xgm2wRljttzRGgq.jpg"
        file = "NdNLO.jpg"

        from pathlib import Path

        img = cv2.imread(str(Path.home() / "Pictures" / file))
        img = to_gray(img)
        ret, img = cv2.threshold(img, 127, 255, 0)
        skel = top_skeleton(img)
        show_image(skel, wait=True)
        cv2.destroyAllWindows()

    def uahsd2():
        # file = "360_F_108702068_Z9VGab1DfiyPzq2v5Xgm2wRljttzRGgq.jpg"
        file = "NdNLO.jpg"

        from pathlib import Path

        img = cv2.imread(str(Path.home() / "Pictures" / file))
        img = to_gray(img)
        ret, img = cv2.threshold(img, 127, 255, 0)
        skel = skeletonise(img)
        show_image(skel, wait=True)
        cv2.destroyAllWindows()

    uahsd2()
