import time

import cv2
from draugr.python_utilities.datetimes import now_repr
from draugr.visualisation.pillow_utilities import pil_image_to_byte_array

from draugr.opencv_utilities import AsyncVideoStream
from mqtt_callbacks import get_mqtt_client
from PIL import Image

from .config import MQTT_CAM_CONFIG

MQTT_BROKER = MQTT_CAM_CONFIG["mqtt"]["broker"]
MQTT_PORT = MQTT_CAM_CONFIG["mqtt"]["port"]
MQTT_QOS = MQTT_CAM_CONFIG["mqtt"]["QOS"]

MQTT_TOPIC_CAMERA = MQTT_CAM_CONFIG["camera"]["mqtt_topic"]
VIDEO_SOURCE = MQTT_CAM_CONFIG["camera"]["video_source"]
FPS = MQTT_CAM_CONFIG["camera"]["fps"]


def main():
  client = get_mqtt_client()
  client.connect(MQTT_BROKER, port=MQTT_PORT)
  time.sleep(4)  # Wait for connection setup to complete
  client.loop_start()

  for frame in AsyncVideoStream(src=VIDEO_SOURCE):
    client.publish(MQTT_TOPIC_CAMERA,
                   pil_image_to_byte_array(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))),
                   qos=MQTT_QOS)
    print(f"published frame on topic: {MQTT_TOPIC_CAMERA} at {now_repr()}")
    time.sleep(1 / FPS)


if __name__ == "__main__":
  main()
