from typing import Optional

import zmq
from draugr.opencv_utilities import show_image
from samples.misc.exclude import SOCKET_ADDRESS2
from samples.misc.opencv_samples.conflated_image_pipe.configuration import ComArchEnum

if __name__ == "__main__":

    def main(
        method: ComArchEnum = ComArchEnum.pubsub, topic: Optional[str] = None
    ) -> None:
        """

        topic: "" is all

        :param method:
        :param topic:
        """
        if topic is None:
            topic = ""  # "" is all

        with zmq.Context() as context:
            with context.socket(method.value["dst"].value) as zmq_socket:
                if method is ComArchEnum.pubsub:
                    zmq_socket.subscribe(topic)
                    # zmq_socket.setsockopt(zmq.ZMQ_RCVHWM, 1)

                zmq_socket.setsockopt(zmq.CONFLATE, 1)
                zmq_socket.setsockopt(zmq.LINGER, 0)

                # zmq_socket.connect(SOCKET_ADDRESS2)
                zmq_socket.bind(SOCKET_ADDRESS2)

                while True:
                    frame = zmq_socket.recv_pyobj(
                        # flags=zmq.NOBLOCK
                    )
                    if show_image(frame):
                        break

    main()
