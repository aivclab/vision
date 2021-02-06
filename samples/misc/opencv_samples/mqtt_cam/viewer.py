import time

import numpy
import streamlit
from draugr.visualisation.pillow_utilities import byte_array_to_pil_image

from mqtt_callbacks import get_mqtt_client
from config import MQTT_CAM_CONFIG

MQTT_BROKER = MQTT_CAM_CONFIG["mqtt"]["broker"]
MQTT_PORT = MQTT_CAM_CONFIG["mqtt"]["port"]
MQTT_QOS = MQTT_CAM_CONFIG["mqtt"]["QOS"]

MQTT_TOPIC = MQTT_CAM_CONFIG["save_captures"]["mqtt_topic"]

VIEWER_WIDTH = 600


def get_random_numpy():
    """Return a dummy frame."""
    return numpy.random.randint(0, 100, size=(32, 32))


title = streamlit.title(MQTT_TOPIC)
viewer = streamlit.image(get_random_numpy(), width=VIEWER_WIDTH)
print(streamlit.info)


# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    streamlit.write(f"Connected with result code {rc} to MQTT broker on {MQTT_BROKER}")


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    if msg.topic != MQTT_TOPIC:
        return
    viewer.image(
        byte_array_to_pil_image(msg.payload).convert("RGB"), width=VIEWER_WIDTH
    )


def main():
    client = get_mqtt_client()
    client.on_message = on_message
    client.connect(MQTT_BROKER, port=MQTT_PORT)
    client.subscribe(MQTT_TOPIC)
    time.sleep(4)  # Wait for connection setup to complete
    client.loop_forever()


if __name__ == "__main__":
    main()
