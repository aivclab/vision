import numpy
import torch
from draugr.torch_utilities import MODULES_MAPPING, get_model_complexity_info

from neodroidvision.classification.architectures.self_attention_network import (
    SelfAttentionTypeEnum,
    make_san,
)
from neodroidvision.classification.architectures.self_attention_network.self_attention_modules.modules import (
    Aggregation,
    Subtraction,
    Subtraction2,
)

if __name__ == "__main__":
    from samples.classification.san.configs.base_san_cfg import SAN_CONFIG


    def main():
        """

        """
        with torch.cuda.device(0):
            def subtraction_flops_counter_hook(module, input, output):
                """

                Args:
                  module:
                  input:
                  output:
                """
                output = output[0]
                module.__flops__ += int(numpy.prod(output.shape))

            def subtraction2_flops_counter_hook(module, input, output):
                """

                Args:
                  module:
                  input:
                  output:
                """
                output = output[0]
                module.__flops__ += int(numpy.prod(output.shape))

            def aggregation_flops_counter_hook(module, input, output):
                """

                Args:
                  module:
                  input:
                  output:
                """
                output = output[0]
                module.__flops__ += int(
                    numpy.prod(output.shape) * numpy.prod(module.kernel_size)
                )

            MODULES_MAPPING[Subtraction] = subtraction_flops_counter_hook
            MODULES_MAPPING[Subtraction2] = subtraction2_flops_counter_hook
            MODULES_MAPPING[Aggregation] = aggregation_flops_counter_hook

            config = SAN_CONFIG

            model = make_san(
                self_attention_type=SelfAttentionTypeEnum(config.self_attention_type),
                layers=config.layers,
                kernels=config.kernels,
                num_classes=config._categories,
            )

            flops, params = get_model_complexity_info(
                model.cuda(), (3, 224, 224), as_strings=True, print_per_layer_stat=True
            )
            print(f"Params/Flops: {params}/{flops}")


    main()
