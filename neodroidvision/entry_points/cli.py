#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/02/2020
           """

import fire
from pyfiglet import Figlet

import draugr
from neodroidvision import get_version

sponsors = "SINTEF Ocean, Alexandra Institute, Norges Forskningsråd"
margin_percentage = 0 / 6
terminal_width = draugr.get_terminal_size().columns
margin = int(margin_percentage * terminal_width)
width = terminal_width - 2 * margin
underline = "_" * width
indent = " " * margin


class NeodroidVisionCLI(object):
    def run(self) -> None:
        pass

    @staticmethod
    def version() -> None:
        """
Prints the version of this Neodroid installation.
"""
        draw_cli_header()
        print(f"Version: {get_version()}")

    @staticmethod
    def sponsors() -> None:
        print(sponsors)


def draw_cli_header(*, title="Neodroid Vision", font="big"):
    figlet = Figlet(font=font, justify="center", width=terminal_width)
    description = figlet.renderText(title)

    print(f"{description}{underline}\n")


def main(*, always_draw_header=False):
    if always_draw_header:
        draw_cli_header()
    fire.Fire(NeodroidVisionCLI, name="neodroid-vision")


if __name__ == "__main__":
    main()
