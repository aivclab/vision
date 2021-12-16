import torch
from torch.nn.functional import cross_entropy, interpolate

__all__ = [
    "cross_entropy2d",
    "multi_scale_cross_entropy2d",
    "bootstrapped_cross_entropy2d",
]


def cross_entropy2d(input, target, weight=None, size_average=True):
    """

    Args:
      input:
      target:
      weight:
      size_average:

    Returns:

    """
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    return cross_entropy(
        input.transpose(1, 2).transpose(2, 3).reshape(-1, c),
        target.reshape(-1),
        weight=weight,
        size_average=size_average,
        ignore_index=250,
    )


def multi_scale_cross_entropy2d(
    input, target, weight=None, size_average=True, scale_weight=None
):
    """

    Args:
      input:
      target:
      weight:
      size_average:
      scale_weight:

    Returns:

    """
    if not isinstance(input, tuple):
        return cross_entropy2d(
            input=input, target=target, weight=weight, size_average=size_average
        )

    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight is None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(
            scale * torch.ones(n_inp), torch.arange(n_inp).float()
        ).to(input.device)

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight, size_average=size_average
        )

    return loss


def bootstrapped_cross_entropy2d(input, target, K: int, weight=None, size_average=True):
    """

    Args:
      input:
      target:
      K:
      weight:
      size_average:

    Returns:

    """
    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, target, K, weight=None, size_average=True):
        n, c, h, w = input.size()
        loss = cross_entropy(
            input.transpose(1, 2).transpose(2, 3).reshape(-1, c),
            target.reshape(-1),
            weight=weight,
            reduce=False,
            size_average=False,
            ignore_index=250,
        )

        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            size_average=size_average,
        )
    return loss / float(batch_size)
