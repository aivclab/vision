from draugr.opencv_utilities import AsyncVideoStream

from draugr.opencv_utilities.dlib_utilities import (
    dlib68FacialLandmarksIndices,
    mouth_aspect_ratio,
    shape_to_ndarray,
)

import dlib
import cv2

if __name__ == "__main__":

    def asd():
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        cv2.namedWindow("test")
        upsample_num_times = 0
        for frame_i, frame in enumerate(AsyncVideoStream()):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for rect_i, rect in enumerate(detector(gray, upsample_num_times)):
                shape = predictor(gray, rect)
                mouth = dlib68FacialLandmarksIndices.slice(
                    shape_to_ndarray(shape), dlib68FacialLandmarksIndices.mouth
                )
                mar = mouth_aspect_ratio(mouth)
                mouth_hull = cv2.convexHull(mouth)
                cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 0), 1)

                # Calibrate needed
                if mar < 0.24:
                    status = "wide smile"
                elif mar < 0.34:
                    status = "neutral"
                elif mar < 0.64:
                    # cv2.imwrite(f"smile_{frame_i}_{rect_i}.png", frame)
                    status = "teeth smile"
                else:
                    status = "yawning"

                cv2.putText(
                    frame,
                    status,
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"mar: {mar}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

    asd()
