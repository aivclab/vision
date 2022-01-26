#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 28/06/2020
           """

import json
from pathlib import Path

if __name__ == "__main__":

    with open(str(Path.home() / "Downloads" / "imagenet_class_index.json")) as f:
        with open("../imagenet_2012_names.py", "w") as sfn:
            with open("../imagenet_2012_id.py", "w") as sfi:
                class_idx = json.load(f)
                sfn.write("categories = {")
                sfi.write("categories = {")
                for k, v in class_idx.items():
                    sfn.write(f'{k}:"{v[1]}",\n')
                    sfi.write(f'{k}:"{v[0]}",\n')
                sfn.write("}")
                sfi.write("}")
