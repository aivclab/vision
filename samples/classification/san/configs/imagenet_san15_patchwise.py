from pathlib import Path

from samples.classification.san.configs.imagenet_san15_pairwise import SAN_CONFIG

SAN_CONFIG.update(
    self_attention_type=1,
    layers=[3, 2, 3, 5, 2],
    kernels=[3, 7, 7, 7, 7],
    save_path=Path("exclude/models/imagenet/san15_patchwise/model"),
    model_path=Path("exclude/models/imagenet/san15_patchwise/model/model_best.pth"),
)
