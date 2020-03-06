from .efficient_net import EfficientNet

__all__ = ["efficient_net_b3", "EfficientNet"]


def efficient_net_b3(cfg, pretrained=True):
    if pretrained:
        model = EfficientNet.from_pretrained("efficientnet-b3")
    else:
        model = EfficientNet.from_name("efficientnet-b3")

    return model
