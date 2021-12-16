import cv2
import zmq
from warg import GDKC, identity

from samples.misc.exclude import SOCKET_ADDRESS1, SOCKET_ADDRESS2
from samples.misc.opencv_samples.conflated_image_pipe.configuration import ComArchEnum


def processor(
    src_socket,
    dst_socket,
    func: GDKC = identity,  # GDKC(cv2.cvtColor, kwargs=dict(code=cv2.COLOR_BGR2GRAY))
):
    """

    :param src_socket:
    :param dst_socket:
    :param func:
    """
    while True:
        dst_socket.send_pyobj(
            func(src_socket.recv_pyobj()),
            copy=False,
            track=False,
            # flags=zmq.NOBLOCK
        )


if __name__ == "__main__":

    def main(method=ComArchEnum.pubsub, topic=""):  # "" is all
        """

        :param method:
        :param topic:
        """
        with zmq.Context() as context:
            with context.socket(method.value[1].value) as src:
                with context.socket(method.value[0].value) as dst:
                    if method is ComArchEnum.pubsub:
                        src.subscribe(topic)
                    src.setsockopt(zmq.CONFLATE, 1)
                    src.setsockopt(zmq.LINGER, 0)
                    dst.setsockopt(zmq.CONFLATE, 1)
                    dst.setsockopt(zmq.LINGER, 0)
                    src.connect(SOCKET_ADDRESS1)
                    dst.connect(SOCKET_ADDRESS2)
                    processor(src, dst)

    main()
