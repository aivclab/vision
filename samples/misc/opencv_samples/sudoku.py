from pathlib import Path

import cv2
from draugr.opencv_utilities import show_image

from neodroidvision.utilities import hough_lines

if __name__ == "__main__":

    def uhasd():
        file = (
            "white-paper-table-13912022.jpg",
            "2.jpg",
            "NdNLO.jpg",
            "sudoku.jpg",
            "sudoku2.jpg",
            "sudoku3.jpg",
        )
        for f in file:
            img = cv2.imread(str(Path.home() / "Pictures" / f))

            lines = hough_lines(img, debug=True)
            for line in lines:  # Draw lines on the image
                x1, y1, x2, y2 = line[0]
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

            if show_image(cv2.resize(img, dsize=(600, 600)), wait=True):
                break

        cv2.destroyAllWindows()

    uhasd()
