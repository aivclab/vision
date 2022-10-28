#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Christian Heider Nielsen"
__doc__ = """ description """

from shutil import rmtree

from warg import IgnoreInterruptSignal


def main(keep_alive: bool = True) -> str:
    """

    Args:
      keep_alive:
    """
    from draugr.torch_utilities import launch_tensorboard
    from time import sleep

    from neodroidvision import PROJECT_APP_PATH

    import argparse

    parser = argparse.ArgumentParser(description="Option for launching tensorboard")
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Adding --clean argument will wipe tensorboard logs",
    )
    parser.add_argument(
        "--port", default=6006, help="Which port should tensorboard be served on"
    )
    args = parser.parse_args()

    log_dir = str(PROJECT_APP_PATH.user_log)

    if args.clean:
        print(f"Wiping {PROJECT_APP_PATH.user_log}")
        if PROJECT_APP_PATH.user_log.exists():
            rmtree(log_dir)
        else:
            PROJECT_APP_PATH.user_log.mkdir()

    address = launch_tensorboard(log_dir, args.port)

    if keep_alive:
        print(f"tensorboard address: {address} for log_dir {log_dir}")
        with IgnoreInterruptSignal():
            while True:
                sleep(100)
    else:
        PROJECT_APP_PATH.user_log.mkdir()

    address = launch_tensorboard(log_dir, args.port)

    if keep_alive:
        print(f"tensorboard address: {address} for log_dir {log_dir}")
        with IgnoreInterruptSignal():
            while True:
                sleep(100)
    else:
        return address


if __name__ == "__main__":
    main()
