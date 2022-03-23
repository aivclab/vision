import cv2
import numpy
from draugr.opencv_utilities import show_image

if __name__ == "__main__":

    img = cv2.imread("home.jpg")
    K = 2
    if img is not None:
        Z = img.reshape((-1, 3))
        Z = numpy.float32(Z)  # convert to numpy.float32

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            10,
            1.0,
        )  # define criteria, number of clusters(K) and apply kmeans()
        ret, label, center = cv2.kmeans(
            Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        center = numpy.uint8(
            center
        )  # Now convert back into uint8, and make original image
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))

        show_image(res2, wait=True)

        cv2.destroyAllWindows()
