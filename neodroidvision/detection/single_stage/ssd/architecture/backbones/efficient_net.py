#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/03/2020
           """

import collections
import re
from typing import List, Tuple
import torch

from torch import nn
from neodroidvision.detection.single_stage.ssd.architecture.backbones.ssd_backbone import (
    SSDBackbone,
)
from neodroidvision.utilities.torch_utilities.efficient_net_utilities import (
    Conv2dSamePadding,
    MobileInvertedResidualBottleneckConvBlock,
    round_filters,
    round_repeats,
)
from neodroidvision.utilities.torch_utilities.output_activation.custom_activations import (
    swish,
)
from neodroidvision.utilities.torch_utilities.persistence.custom_model_caching import (
    load_state_dict_from_url,
)

__all__ = ["EfficientNet"]


class EfficientNet(SSDBackbone):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
    blocks_args (list): A list of BlockArgs to construct blocks
    global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
    model = EfficientNet.from_pretrained('efficientnet-b0')
    """

    @staticmethod
    def add_extras(cfgs):
        """

        :param cfgs:
        :type cfgs:
        :return:
        :rtype:"""
        extras = nn.ModuleList()
        for cfg in cfgs:
            extra = []
            for params in cfg:
                in_channels, out_channels, kernel_size, stride, padding = params
                extra.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
                )
                extra.append(nn.ReLU())
            extras.append(nn.Sequential(*extra))
        return extras

    INDICES = {"efficientnet-b3": [7, 17, 25]}

    EXTRAS = {
        "efficientnet-b3": [
            # in,  out, k, s, p
            [(384, 128, 1, 1, 0), (128, 256, 3, 2, 1)],  # 5 x 5
            [(256, 128, 1, 1, 0), (128, 256, 3, 1, 0)],  # 3 x 3
            [(256, 128, 1, 1, 0), (128, 256, 3, 1, 0)],  # 1 x 1
        ]
    }

    def __init__(self, size, model_name, blocks_args=None, global_params=None):
        super().__init__(size)
        self.indices = self.INDICES[model_name]
        self.extras = self.add_extras(self.EXTRAS[model_name])
        assert isinstance(blocks_args, list), "blocks_args should be a list"
        assert len(blocks_args) > 0, "block args must be greater than 0"
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(
            32, self._global_params
        )  # number of output channels
        self._conv_stem = Conv2dSamePadding(
            in_channels, out_channels, kernel_size=3, stride=2, bias=False
        )
        self._bn0 = nn.BatchNorm2d(
            num_features=out_channels, momentum=bn_mom, eps=bn_eps
        )

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(
                    block_args.input_filters, self._global_params
                ),
                output_filters=round_filters(
                    block_args.output_filters, self._global_params
                ),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params),
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(
                MobileInvertedResidualBottleneckConvBlock(
                    block_args, self._global_params
                )
            )
            if block_args.num_repeat > 1:
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, stride=1
                )
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(
                    MobileInvertedResidualBottleneckConvBlock(
                        block_args, self._global_params
                    )
                )
        self.reset_parameters()

    def extract_features(self, inputs: torch.Tensor) -> Tuple:
        """Returns output of the final convolution layer"""

        # Stem
        x = swish(self._bn0(self._conv_stem(inputs)))

        features = []

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate)
            if idx in self.indices:
                features.append(x)

        return x, features

    def forward(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        """Calls extract_features to extract features, applies final linear layer, and returns logits."""

        # Convolution layers
        x, features = self.extract_features(inputs)

        for layer in self.extras:
            x = layer(x)
            features.append(x)

        return features

    @staticmethod
    def efficientnet_params(model_name: str) -> Tuple[float, float, int, float]:
        """Map EfficientNet model name to parameter coefficients."""
        params_dict = {
            # Coefficients:   width,depth,res,dropout
            "efficientnet-b0": (1.0, 1.0, 224, 0.2),
            "efficientnet-b1": (1.0, 1.1, 240, 0.2),
            "efficientnet-b2": (1.1, 1.2, 260, 0.3),
            "efficientnet-b3": (1.2, 1.4, 300, 0.3),
            "efficientnet-b4": (1.4, 1.8, 380, 0.4),
            "efficientnet-b5": (1.6, 2.2, 456, 0.4),
            "efficientnet-b6": (1.8, 2.6, 528, 0.5),
            "efficientnet-b7": (2.0, 3.1, 600, 0.5),
        }
        return params_dict[model_name]

    @staticmethod
    def interpret(
        width_coefficient=None,
        depth_coefficient=None,
        dropout_rate=0.2,
        drop_connect_rate=0.2,
    ):
        """Creates a efficient net model."""

        GlobalParams = collections.namedtuple(
            "GlobalParams",
            [
                "batch_norm_momentum",
                "batch_norm_epsilon",
                "dropout_rate",
                "num_categories",
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
            """Block Decoder for readability, straight from the official TensorFlow repository"""

            @staticmethod
            def _decode_block_string(block_string):
                """Gets a block through a string notation of arguments."""
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
                    f"r{block.num_repeat:d}",
                    f"k{block.kernel_size:d}",
                    f"s{block.strides[0]:d}{block.strides[1]:d}",
                    f"e{block.expand_ratio}",
                    f"i{block.input_filters:d}",
                    f"o{block.output_filters:d}",
                ]
                if 0 < block.se_ratio <= 1:
                    args.append(f"se{block.se_ratio}")
                if block.id_skip is False:
                    args.append("noskip")
                return "_".join(args)

            @staticmethod
            def decode(string_list):
                """
                Decodes a list of string notations to specify blocks inside the network.

                :param string_list: a list of strings, each string is a notation of block
                :return: a list of BlockArgs namedtuples of block args"""
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
                :return: a list of strings, each string is a notation of block"""
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
            num_categories=1000,
            width_coefficient=width_coefficient,
            depth_coefficient=depth_coefficient,
            depth_divisor=8,
            min_depth=None,
        )

        return blocks_args, global_params

    @staticmethod
    def get_model_params(model_name, override_params):
        """Get the block args and global params for a given model"""
        if model_name.startswith("efficientnet"):
            w, d, _, p = EfficientNet.efficientnet_params(model_name)
            # note: all models have drop connect rate = 0.2
            blocks_args, global_params = EfficientNet.interpret(
                width_coefficient=w, depth_coefficient=d, dropout_rate=p
            )
        else:
            raise NotImplementedError(f"model name is not pre-defined: {model_name}")
        if override_params:
            # ValueError will be raised here if override_params has fields not included in global_params.
            global_params = global_params._replace(**override_params)
        return blocks_args, global_params

    @classmethod
    def from_name(cls, model_name, override_params=None):
        """

        :param model_name:
        :type model_name:
        :param override_params:
        :type override_params:
        :return:
        :rtype:"""
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = EfficientNet.get_model_params(
            model_name, override_params
        )
        return EfficientNet(0, model_name, blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name):
        """
        Loads pretrained weights, and downloads if loading for the first time.

        :param model_name:
        :type model_name:
        :return:
        :rtype:"""

        model = EfficientNet.from_name(model_name)

        url_map = {
            "efficientnet-b0": "http://storage.googleapis.com/public-models/efficientnet-b0-08094119.pth",
            "efficientnet-b1": "http://storage.googleapis.com/public-models/efficientnet-b1-dbc7070a.pth",
            "efficientnet-b2": "http://storage.googleapis.com/public-models/efficientnet-b2-27687264.pth",
            "efficientnet-b3": "http://storage.googleapis.com/public-models/efficientnet-b3-c8376fa2.pth",
            "efficientnet-b4": "http://storage.googleapis.com/public-models/efficientnet-b4-e116e8b3.pth",
            "efficientnet-b5": "http://storage.googleapis.com/public-models/efficientnet-b5-586e6cc6.pth",
        }

        state_dict = load_state_dict_from_url(url_map[model_name])
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights for {model_name}")

        return model

    @classmethod
    def get_image_size(cls, model_name):
        """

        :param model_name:
        :type model_name:
        :return:
        :rtype:"""
        cls._check_model_name_is_valid(model_name)
        *_, res, _ = EfficientNet.efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        """Validates model name. None that pretrained weights are only available for
        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment."""
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = [f"efficientnet_b{str(i)}" for i in range(num_models)]
        if model_name.replace("-", "_") not in valid_models:
            raise ValueError(f"model_name should be one of: {', '.join(valid_models)}")
