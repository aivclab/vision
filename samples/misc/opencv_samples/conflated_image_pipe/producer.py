import zmq
from draugr.opencv_utilities import AsyncVideoStream

from samples.misc.exclude import SOCKET_ADDRESS1
from samples.misc.opencv_samples.conflated_image_pipe.configuration import ComArchEnum

if __name__ == "__main__":

    def main(method=ComArchEnum.pubsub):
        """

        :param method:
        """
        with zmq.Context() as context:  # TODO: maybe set zmq.AFFINITY
            with context.socket(method.value[0].value) as socket:
                # dst = context.socket(zmq.PUB) # fine-tune zmq.SNDBUF + zmq.SNDHWM on PUB side, if multi-subscribers are expected and zmq.CONFLATE is not used.
                # dst.setsockopt(zmq.SNDBUF, 1)
                # dst.setsockopt(zmq.ZMQ_SNDHWM, 1)

                socket.setsockopt(zmq.CONFLATE, 1)
                socket.setsockopt(zmq.LINGER, 0)

                socket.bind(SOCKET_ADDRESS1)

                for frame in AsyncVideoStream():
                    socket.send_pyobj(
                        frame,
                        copy=False,
                        track=False,
                        # flags=zmq.NOBLOCK
                    )

    main()
