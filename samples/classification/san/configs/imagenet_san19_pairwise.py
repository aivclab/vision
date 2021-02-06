from pathlib import Path

from samples.classification.san.configs.base_san_cfg import SAN_CONFIG

SAN_CONFIG.update(
    self_attention_type=0,
    layers=[3, 3, 4, 6, 3],
    kernels=[3, 7, 7, 7, 7],
    ignore_label=2000,
    base_lr=0.1,
    epochs=100,
    start_epoch=0,
    step_epochs=[30, 60, 90],
    label_smoothing=0.1,
    momentum=0.9,
    weight_decay=0.0001,
    save_path=Path("exclude/models/imagenet/san19_pairwise/model"),
    model_path=Path("exclude/models/imagenet/san19_pairwise/model/model_best.pth"),
)
