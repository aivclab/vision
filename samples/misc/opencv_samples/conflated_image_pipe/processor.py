import zmq
from samples.misc.exclude import SOCKET_ADDRESS1, SOCKET_ADDRESS2
from samples.misc.opencv_samples.conflated_image_pipe.configuration import ComArchEnum
from typing import Optional
from warg import GDKC, identity
from zmq import Socket


def processor(
    src_socket: Socket,
    dst_socket: Socket,
    func: GDKC = identity,  # GDKC(cv2.cvtColor, kwargs=dict(code=cv2.COLOR_BGR2GRAY))
) -> None:
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

    def main(
        method: ComArchEnum = ComArchEnum.pubsub, topic: Optional[str] = None
    ) -> None:  # "" is all
        """

        topic: "" is all

        :param method:
        :param topic:
        """
        if topic is None:
            topic = ""  # "" is all
        with zmq.Context() as context:
            with context.socket(method.value["dst"].value) as src:
                with context.socket(method.value["src"].value) as dst:
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
