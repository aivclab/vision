from typing import Tuple

import torch
from torch import nn

from . import ram_modules

__author__ = "Christian"
__doc__ = "Foveal attention, moves around gaze and yields glimpses"


class RecurrentAttention(nn.Module):
    """A Recurrent Model of Visual Attention (RAM) [1].

    RAM is a recurrent neural network that processes
    inputs sequentially, attending to different locations
    within the image one at a time, and incrementally
    combining information from these fixations to build
    up a dynamic internal representation of the image.

    References:
    [1]: Minh et. al., https://arxiv.org/abs/1406.6247"""

    def __init__(
        self,
        size_glimpse,
        num_patches_per_glimpse,
        scale_factor_suc,
        num_channels,
        hidden_size_glimpse,
        hidden_size_locator,
        std_policy,
        hidden_size_rnn,
        num_classes,
    ):
        """Constructor.

        Args:
        size_glimpse: size of the square patches in the glimpses extracted by the retina.
        num_patches_per_glimpse: number of patches to extract per glimpse.
        scale_factor_suc: scaling factor that controls the size of successive patches.
        num_channels: number of channels in each image.
        hidden_size_glimpse: hidden layer size of the fc layer for `phi`.
        hidden_size_locator: hidden layer size of the fc layer for `l`.
        std_policy: standard deviation of the Gaussian policy.
        hidden_size_rnn: hidden size of the rnn.
        num_classes: number of classes in the dataset.
        num_glimpses: number of glimpses to take per image,
        i.e. number of BPTT steps."""
        super().__init__()

        self._sensor = ram_modules.GlimpseSensor(
            hidden_size_glimpse,
            hidden_size_locator,
            size_glimpse,
            num_patches_per_glimpse,
            scale_factor_suc,
            num_channels,
        )
        self._rnn = ram_modules.CoreRNN(hidden_size_rnn, hidden_size_rnn)
        self._locator_policy = ram_modules.Locator(hidden_size_rnn, 2, std_policy)
        self.classifier = ram_modules.Actor(hidden_size_rnn, num_classes)
        self._signal_baseline = ram_modules.SignalBaseline(hidden_size_rnn, 1)

    def forward(
        self,
        x: torch.Tensor,
        l_t_prev: torch.Tensor,
        h_t_prev: torch.Tensor,
        last: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Run RAM for one step on a minibatch of images.

        Args:
        x: a 4D Tensor of shape (B, H, W, C). The minibatch
            of images.
        l_t_prev: a 2D tensor of shape (B, 2). The location vector
            containing the glimpse coordinates [x, y] for the previous
            step `t-1`.
        h_t_prev: a 2D tensor of shape (B, hidden_size). The hidden
            state vector for the previous step `t-1`.
        last: a bool indicating whether this is the last step.
            If True, the action network returns an output probability
            vector over the classes and the baseline `b_t` for the
            current step `t`. Else, the core network returns the
            hidden state vector for the next step `t+1` and the
            location vector for the next step `t+1`.

        Returns:
        h_t: a 2D tensor of shape (B, hidden_size). The hidden
            state vector for the current step `t`.
        mu: a 2D tensor of shape (B, 2). The mean that parametrizes
            the Gaussian policy.
        l_t: a 2D tensor of shape (B, 2). The location vector
            containing the glimpse coordinates [x, y] for the
            current step `t`.
        b_t: a vector of length (B,). The baseline for the
            current time step `t`.
        log_probas: a 2D tensor of shape (B, num_classes). The
            output log probability vector over the classes.
        log_pi: a vector of length (B,)."""
        h_t = self._rnn(self._sensor(x, l_t_prev), h_t_prev)

        log_pi, l_t = self._locator_policy(h_t)
        b_t = self._signal_baseline(h_t).squeeze()

        if last:
            return (h_t, l_t, b_t, self.classifier(h_t), log_pi)  # log_probas

        return h_t, l_t, b_t, log_pi
