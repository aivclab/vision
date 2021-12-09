import time
from draugr.python_utilities.datetimes import now_repr
from draugr.visualisation.pillow_utilities import byte_array_to_pil_image

from mqtt_callbacks import get_mqtt_client
from .config import MQTT_CAM_CONFIG

MQTT_BROKER = MQTT_CAM_CONFIG["mqtt"]["broker"]
MQTT_PORT = MQTT_CAM_CONFIG["mqtt"]["port"]
MQTT_QOS = MQTT_CAM_CONFIG["mqtt"]["QOS"]

SAVE_TOPIC = MQTT_CAM_CONFIG["save-captures"]["mqtt_topic"]
CAPTURES_DIRECTORY = MQTT_CAM_CONFIG["save-captures"]["captures_directory"]


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
        image = byte_array_to_pil_image(msg.payload).convert("RGB")

        save_file_path = CAPTURES_DIRECTORY + f"capture_{now}.jpg"
        image.save(save_file_path)
        print(f"Saved {save_file_path}")

    except Exception as exc:
        print(exc)


def main():
    """

    """
    client = get_mqtt_client()
    client.on_message = on_message
    client.connect(MQTT_BROKER, port=MQTT_PORT)
    client.subscribe(SAVE_TOPIC)
    time.sleep(4)  # Wait for connection setup to complete
    client.loop_forever()


if __name__ == "__main__":
    main()
