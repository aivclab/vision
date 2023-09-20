# !/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
           """

import time
from config import MQTT_CAM_CONFIG
from draugr.visualisation.pillow_utilities import (
    byte_array_to_pil_image,
    pil_image_to_byte_array,
)
from warg import now_repr

from .mqtt_callbacks import get_mqtt_client

MQTT_BROKER = MQTT_CAM_CONFIG["mqtt"]["broker"]
MQTT_PORT = MQTT_CAM_CONFIG["mqtt"]["port"]
MQTT_QOS = MQTT_CAM_CONFIG["mqtt"]["QOS"]

MQTT_SUBSCRIBE_TOPIC = MQTT_CAM_CONFIG["processing"]["subscribe_topic"]
MQTT_PUBLISH_TOPIC = MQTT_CAM_CONFIG["processing"]["publish_topic"]

ROTATE_ANGLE = 45  # Angle of rotation in degrees to apply


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    """

    Args:
      client:
      userdata:
      msg:
    """
    now = now_repr()
    print("message on " + str(msg.topic) + f" at {now}")
    try:
        client.publish(
            MQTT_PUBLISH_TOPIC,
            pil_image_to_byte_array(byte_array_to_pil_image(msg.payload).rotate()),
            qos=MQTT_QOS,
        )
        print(f"published processed frame on topic: {MQTT_PUBLISH_TOPIC} at {now}")

    except Exception as exc:
        print(exc)


def main():
    """description"""
    client = get_mqtt_client()
    client.on_message = on_message
    client.connect(MQTT_BROKER, port=MQTT_PORT)
    client.subscribe(MQTT_SUBSCRIBE_TOPIC)
    time.sleep(4)  # Wait for connection setup to complete
    client.loop_forever()


if __name__ == "__main__":
    main()
