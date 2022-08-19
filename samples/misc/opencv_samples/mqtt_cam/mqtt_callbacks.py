#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
           """

from paho.mqtt import client


def on_connect(c, userdata, flags, rc):
    """

    Args:
      c:
      userdata:
      flags:
      rc:
    """
    # print(f"CONNACK received with code {rc}")
    if rc == 0:
        print("connected to MQTT broker")
        c.connected_flag = True  # set flag
    else:
        print("Bad connection to MQTT broker, returned code=", rc)


def on_publish(c, userdata, mid):
    """

    Args:
      c:
      userdata:
      mid:
    """
    print(f"mid: {str(mid)}")


def get_mqtt_client():
    """Return the MQTT client object."""
    c = client.Client()
    c.connected_flag = False  # set flag
    c.on_connect = on_connect
    c.on_publish = on_publish
    return c
