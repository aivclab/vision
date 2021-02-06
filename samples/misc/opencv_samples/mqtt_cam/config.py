from pathlib import Path

from warg import NOD

MQTT_CAM_CONFIG = NOD(
    mqtt=NOD(
        broker="localhost", port=1883, QOS=1  # or an ip address like 192.168.1.74
    ),
    camera=NOD(
        video_source=0,
        fps=30,  # 2
        mqtt_topic="video/video0/capture",
        # If your desired camera is listed as source 0 you will configure video_source: 0. Alternatively you can configure the video source as an MJPEG or RTSP stream. For example in config.yml you may configure something like video_source: "rtsp://admin:password@192.168.1.94:554/11" for a RTSP camera.
    ),
    processing=NOD(
        subscribe_topic="video/video0/capture",
        publish_topic="video/video0/capture/rotated",
    ),
    save_captures=NOD(
        mqtt_topic="video/video0/capture", captures_directory=Path("captures")
    ),
)
