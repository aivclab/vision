from pathlib import Path

from samples.classification.san.configs.imagenet_san10_pairwise import SAN_CONFIG

SAN_CONFIG.update(
    self_attention_type=1,
    save_path=Path("exclude/models/imagenet/san10_patchwise/model"),
    model_path=Path("exclude/models/imagenet/san10_patchwise/model/model_best.pth"),
    )
