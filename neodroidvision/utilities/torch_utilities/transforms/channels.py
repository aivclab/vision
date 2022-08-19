from torchvision import transforms

__all__ = ["Replicate"]

Replicate = lambda n: transforms.Lambda(lambda x: x.repeat(n, 1, 1))
