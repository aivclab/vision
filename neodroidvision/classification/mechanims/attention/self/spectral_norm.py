from torch import nn
from torch.nn.utils import spectral_norm


def spectral_norm_conv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
):
    """

    Args:
      in_channels:
      out_channels:
      kernel_size:
      stride:
      padding:
      dilation:
      groups:
      bias:

    Returns:

    """
    return spectral_norm(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
    )


def spectral_norm_linear(in_features, out_features):
    """

    Args:
      in_features:
      out_features:

    Returns:

    """
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features))


def spectral_norm_embedding(num_embeddings, embedding_dim):
    """

    Args:
      num_embeddings:
      embedding_dim:

    Returns:

    """
    return spectral_norm(
        nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    )
