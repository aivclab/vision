from enum import Enum

import zmq

__all__ = ['ReceiveMethodEnum', 'SendMethodEnum', 'ComArchEnum']


class ReceiveMethodEnum(Enum):
  '''

  '''
  pull = zmq.PULL
  sub = zmq.SUB


class SendMethodEnum(Enum):
  '''

  '''
  push = zmq.PUSH
  pub = zmq.PUB


class ComArchEnum(Enum):
  '''

  '''
  pubsub = (SendMethodEnum.pub, ReceiveMethodEnum.sub)
  pushpull = (SendMethodEnum.push, ReceiveMethodEnum.pull)
