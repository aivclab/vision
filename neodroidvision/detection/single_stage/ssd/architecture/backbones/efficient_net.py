from torch import nn

from neodroidvision.detection.single_stage.ssd.architecture.backbones.efficient_net_utilities import (
    Conv2dSamePadding,
    MobileInvertedResidualBottleneckConvBlock,
    efficientnet,
    round_filters,
    round_repeats,
    swish,
)
from neodroidvision.detection.single_stage.ssd.architecture.backbones.ssd_backbone import (
    SSDBackbone,
)
from neodroidvision.utilities.torch_utilities.custom_model_caching import (
    load_state_dict_from_url,
)


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

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

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

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """

        # Convolution layers
        x, features = self.extract_features(inputs)

        for layer in self.extras:
            x = layer(x)
            features.append(x)

        return tuple(features)

    @staticmethod
    def efficientnet_params(model_name):
        """ Map EfficientNet model name to parameter coefficients. """
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
    def get_model_params(model_name, override_params):
        """ Get the block args and global params for a given model """
        if model_name.startswith("efficientnet"):
            w, d, _, p = EfficientNet.efficientnet_params(model_name)
            # note: all models have drop connect rate = 0.2
            blocks_args, global_params = efficientnet(
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
    :rtype:
    """

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
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = EfficientNet.efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        """ Validates model name. None that pretrained weights are only available for
the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ["efficientnet_b" + str(i) for i in range(num_models)]
        if model_name.replace("-", "_") not in valid_models:
            raise ValueError(f"model_name should be one of: {', '.join(valid_models)}")
