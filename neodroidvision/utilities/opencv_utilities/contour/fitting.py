import cv2

__all__ = ["fit_line", "fit_ellipse"]


def fit_ellipse(img, cnt):
    ellipse = cv2.fitEllipse(cnt)
    cv2.ellipse(img, ellipse, (0, 255, 0), 2)


def fit_line(img, cnt):
    rows, cols = img.shape[:2]
    [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    cv2.line(img, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
