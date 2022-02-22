import cv2
import numpy


def top_skelet():
    img = cv2.imread("sofsk.png", 0)
    size = numpy.size(img)
    skel = numpy.zeros(img.shape, numpy.uint8)

    ret, img = cv2.threshold(img, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    cv2.imshow("skel", skel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def skeletise(image):
    return cv2.ximgproc.thinning(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
