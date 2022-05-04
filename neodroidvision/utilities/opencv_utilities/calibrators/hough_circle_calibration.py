from pathlib import Path

import cv2
from draugr.opencv_utilities import show_image, add_trackbar


def hough_circle_calibrator(
    frame,
) -> None:  # TODO: GENERALISE INTERACTIVE CALIBRATOR TO MANY MORE OPENCV FUNCTIONS
    """ """
    a_key_pressed = None  # .init
    edges = None

    lo_label = "Lo_Threshold"
    lo = 127
    lo_prev = -1

    hi_label = "Hi_Threshold"
    hi = 255
    hi_prev = -1

    dp_label = "dp"
    dp = 1
    dp_prev = -1

    min_distance_label = "min_distance"
    min_distance = 10
    min_distance_prev = -1

    param1_label = "param1"
    param1 = 255
    param1_prev = -1

    param2_label = "param2"
    param2 = 20
    param2_prev = -1

    min_radius_label = "min_radius"
    min_radius = 10
    min_radius_prev = -1

    max_radius_label = "max_radius"
    max_radius = 30
    max_radius_prev = -1

    frame_window_label = "Frame"
    canny_frame_window_label = f"{frame_window_label}.Canny"
    canny_hough_circle_window_label = f"{canny_frame_window_label}.Circles"

    show_image(frame, frame_window_label)

    show_image(frame, canny_frame_window_label)
    add_trackbar(canny_frame_window_label, lo_label, default=lo, max_val=1000)
    add_trackbar(canny_frame_window_label, hi_label, default=hi, max_val=1000)

    show_image(frame, canny_hough_circle_window_label)
    add_trackbar(
        canny_hough_circle_window_label, dp_label, default=dp, min_val=1, max_val=20
    )
    add_trackbar(
        canny_hough_circle_window_label,
        min_distance_label,
        default=min_distance,
        min_val=1,
        max_val=1000,
    )
    add_trackbar(
        canny_hough_circle_window_label,
        param1_label,
        default=param1,
        min_val=1,  # Zero is not a valid value for this parameter
    )
    add_trackbar(
        canny_hough_circle_window_label,
        param2_label,
        default=param2,
        min_val=1,  # Zero is not a valid value for this parameter
    )
    add_trackbar(
        canny_hough_circle_window_label, min_radius_label, default=min_radius, min_val=1
    )
    add_trackbar(
        canny_hough_circle_window_label, max_radius_label, default=max_radius, min_val=1
    )

    print(
        " --------------------------------------------------------------------------- press [ESC] to exit "
    )
    while True:
        if a_key_pressed == 27:
            break

        lo = cv2.getTrackbarPos(lo_label, canny_frame_window_label)
        hi = cv2.getTrackbarPos(hi_label, canny_frame_window_label)

        if lo != lo_prev or hi != hi_prev:  # --------------------------= RE-SYNC

            a_canny_refresh_flag = True  # --------------------------= FLAG

            lo_prev = lo
            hi_prev = hi
        else:

            a_canny_refresh_flag = False  # --------------------------= Un-FLAG

        dp = cv2.getTrackbarPos(dp_label, canny_hough_circle_window_label)
        min_distance = cv2.getTrackbarPos(
            min_distance_label, canny_hough_circle_window_label
        )
        param1 = cv2.getTrackbarPos(param1_label, canny_hough_circle_window_label)
        param2 = cv2.getTrackbarPos(param2_label, canny_hough_circle_window_label)
        min_radius = cv2.getTrackbarPos(
            min_radius_label, canny_hough_circle_window_label
        )
        max_radius = cv2.getTrackbarPos(
            max_radius_label, canny_hough_circle_window_label
        )

        if (
            dp != dp_prev
            or min_distance != min_distance_prev
            or param1 != param1_prev
            or param2 != param2_prev
            or min_radius != min_radius_prev
            or max_radius != max_radius_prev
        ):  # ----------------------------------------------= RE-SYNC

            a_hough_refresh_flag = True  # --------------------------= FLAG

            dp_prev = dp
            min_distance_prev = min_distance
            param1_prev = param1
            param2_prev = param2
            min_radius_prev = min_radius
            max_radius_prev = max_radius
        else:

            a_hough_refresh_flag = False  # --------------------------= Un-FLAG

        if (
            a_canny_refresh_flag
        ):  # REFRESH-process-pipe-line ( with recent <state> <vars> )
            edges = cv2.Canny(frame, lo, hi)

            cv2.imshow(canny_frame_window_label, edges)

        if a_canny_refresh_flag or a_hough_refresh_flag:

            circles = cv2.HoughCircles(
                edges,
                cv2.HOUGH_GRADIENT,
                dp=dp,
                minDist=min_distance,
                param1=param1,
                param2=param2,
                minRadius=min_radius,
                maxRadius=max_radius,
            )

            demo_with_circles = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )  # .re-init <<< src
            demo_with_circles = cv2.cvtColor(demo_with_circles, cv2.COLOR_RGB2BGR)
            if circles is not None:
                for c in circles[0]:
                    cv2.circle(
                        demo_with_circles,
                        (int(c[0]), int(c[1])),
                        int(c[2]),
                        (0, 255, 0),
                        1,
                    )

            cv2.imshow(canny_hough_circle_window_label, demo_with_circles)

        a_key_pressed = cv2.waitKey(1) & 0xFF

    cv2.destroyWindow(frame_window_label)
    cv2.destroyWindow(canny_frame_window_label)
    cv2.destroyWindow(canny_hough_circle_window_label)

    # cv2.destroyAllWindows()


if __name__ == "__main__":

    def ijasd():
        from draugr.opencv_utilities import clean_up

        orig = cv2.imread(
            str(Path(r"C:\Users\Christian\OneDrive\Billeder\buh\7BIsT.png"))
        )[:800, :800, :]
        cleaned = clean_up(orig)
        hough_circle_calibrator(cleaned)

    ijasd()
