#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/03/2020
           """

import collections
import re


def efficientnet(
    width_coefficient=None,
    depth_coefficient=None,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
):
    """ Creates a efficientnet model. """

    GlobalParams = collections.namedtuple(
        "GlobalParams",
        [
            "batch_norm_momentum",
            "batch_norm_epsilon",
            "dropout_rate",
            "num_classes",
            "width_coefficient",
            "depth_coefficient",
            "depth_divisor",
            "min_depth",
            "drop_connect_rate",
        ],
    )

    # Parameters for an individual model block
    BlockArgs = collections.namedtuple(
        "BlockArgs",
        [
            "kernel_size",
            "num_repeat",
            "input_filters",
            "output_filters",
            "expand_ratio",
            "id_skip",
            "stride",
            "se_ratio",
        ],
    )

    # Change namedtuple defaults
    GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
    BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

    class BlockDecoder(object):
        """ Block Decoder for readability, straight from the official TensorFlow repository """

        @staticmethod
        def _decode_block_string(block_string):
            """ Gets a block through a string notation of arguments. """
            assert isinstance(block_string, str)

            ops = block_string.split("_")
            options = {}
            for op in ops:
                splits = re.split(r"(\d.*)", op)
                if len(splits) >= 2:
                    key, value = splits[:2]
                    options[key] = value

            # Check stride
            assert ("s" in options and len(options["s"]) == 1) or (
                len(options["s"]) == 2 and options["s"][0] == options["s"][1]
            )

            return BlockArgs(
                kernel_size=int(options["k"]),
                num_repeat=int(options["r"]),
                input_filters=int(options["i"]),
                output_filters=int(options["o"]),
                expand_ratio=int(options["e"]),
                id_skip=("noskip" not in block_string),
                se_ratio=float(options["se"]) if "se" in options else None,
                stride=[int(options["s"][0])],
            )

        @staticmethod
        def _encode_block_string(block):
            """Encodes a block to a string."""
            args = [
                "r%d" % block.num_repeat,
                "k%d" % block.kernel_size,
                "s%d%d" % (block.strides[0], block.strides[1]),
                "e%s" % block.expand_ratio,
                "i%d" % block.input_filters,
                "o%d" % block.output_filters,
            ]
            if 0 < block.se_ratio <= 1:
                args.append("se%s" % block.se_ratio)
            if block.id_skip is False:
                args.append("noskip")
            return "_".join(args)

        @staticmethod
        def decode(string_list):
            """
  Decodes a list of string notations to specify blocks inside the network.

  :param string_list: a list of strings, each string is a notation of block
  :return: a list of BlockArgs namedtuples of block args
  """
            assert isinstance(string_list, list)
            blocks_args = []
            for block_string in string_list:
                blocks_args.append(BlockDecoder._decode_block_string(block_string))
            return blocks_args

        @staticmethod
        def encode(blocks_args):
            """
  Encodes a list of BlockArgs to a list of strings.

  :param blocks_args: a list of BlockArgs namedtuples of block args
  :return: a list of strings, each string is a notation of block
  """
            block_strings = []
            for block in blocks_args:
                block_strings.append(BlockDecoder._encode_block_string(block))
            return block_strings

    blocks_args = [
        "r1_k3_s11_e1_i32_o16_se0.25",
        "r2_k3_s22_e6_i16_o24_se0.25",
        "r2_k5_s22_e6_i24_o40_se0.25",
        "r3_k3_s22_e6_i40_o80_se0.25",
        "r3_k5_s11_e6_i80_o112_se0.25",
        "r4_k5_s22_e6_i112_o192_se0.25",
        "r1_k3_s11_e6_i192_o320_se0.25",
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        # data_format='channels_last',  # removed, this is always true in PyTorch
        num_classes=1000,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
    )

    return blocks_args, global_params
