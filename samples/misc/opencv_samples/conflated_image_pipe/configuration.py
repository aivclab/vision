import zmq
from enum import Enum

__all__ = ["ReceiveMethodEnum", "SendMethodEnum", "ComArchEnum"]


class ReceiveMethodEnum(Enum):
    """
    Supported receive methods
    """

    pull = zmq.PULL
    sub = zmq.SUB


class SendMethodEnum(Enum):
    """
    Supported send methods
    """

    push = zmq.PUSH
    pub = zmq.PUB


class ComArchEnum(Enum):
    """
    Supported Communication methods
    """

    pubsub = {"src": SendMethodEnum.pub, "dst": ReceiveMethodEnum.sub}
    pushpull = {"src": SendMethodEnum.push, "dst": ReceiveMethodEnum.pull}
