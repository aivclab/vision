#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 11/14/22
           """

__all__ = []

from neodroidvision.data.synthesis import resize_children


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Option for batch resizing files in dir"
    )

    parser.add_argument("--src", type=str, help="Source directory", required=True)
    parser.add_argument("--size", type=int, help="Size of images", default=512)
    parser.add_argument(
        "--dst",
        type=str,
        help="Destination directory",
        default="resized",
    )

    args = parser.parse_args()

    resize_children(args.src, args.size, args.dst)


if __name__ == "__main__":
    main()
