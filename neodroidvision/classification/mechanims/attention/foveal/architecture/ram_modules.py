from typing import Tuple

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional


class GlimpseSensor(nn.Module):
    """The glimpse network.

    Combines the "what" and the "where" into a glimpse
    feature vector `g_t`.

    - "what": glimpse extracted from the retina.
    - "where": location tuple where glimpse was extracted.

    Concretely, feeds the output of the retina `phi` to
    a fc layer and the glimpse location vector `l_t_prev`
    to a fc layer. Finally, these outputs are fed each
    through a fc layer and their sum is rectified.

    In other words:

      `g_t = relu( fc( fc(l) ) + fc( fc(phi) ) )`

    Args:
      h_g: hidden layer size of the fc layer for `phi`.
      h_l: hidden layer size of the fc layer for `l`.
      g: size of the square patches in the glimpses extracted
      by the retina.
      k: number of patches to extract per glimpse.
      s: scaling factor that controls the size of successive patches.
      c: number of channels in each image.
      x: a 4D Tensor of shape (B, H, W, C). The minibatch
          of images.
      l_t_prev: a 2D tensor of shape (B, 2). Contains the glimpse
          coordinates [x, y] for the previous timestep `t-1`.

    Returns:
      g_t: a 2D tensor of shape (B, hidden_size).
          The glimpse representation returned by
          the glimpse network for the current
          timestep `t`."""

    class Retina:
        """A visual retina.

        Extracts a foveated glimpse `phi` around location `l`
        from an image `x`.

        Concretely, encodes the region around `l` at a
        high-resolution but uses a progressively lower
        resolution for pixels further from `l`, resulting
        in a compressed representation of the original
        image `x`.

        Args:
        x: a 4D Tensor of shape (B, H, W, C). The minibatch
            of images.
        l: a 2D Tensor of shape (B, 2). Contains normalized
            coordinates in the range [-1, 1].
        size_first_patch: size of the first square patch.
        num_patches_per_glimpse: number of patches to extract in the glimpse.
        scale_factor_suc: scaling factor that controls the size of
            successive patches.

        Returns:
        phi: a 5D tensor of shape (B, k, g, g, C). The
            foveated glimpse of the image."""

        def __init__(self, size_first_patch, num_patches_per_glimpse, scale_factor_suc):
            self.g = size_first_patch
            self.k = num_patches_per_glimpse
            self.s = scale_factor_suc

        def foveate(self, x: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
            """Extract `k` square patches of size `g`, centered
            at location `l`. The initial patch is a square of
            size `g`, and each subsequent patch is a square
            whose side is `s` times the size of the previous
            patch.

            The `k` patches are finally resized to (g, g) and
            concatenated into a tensor of shape (B, k, g, g, C)."""
            phi = []
            size = self.g

            # extract k patches of increasing size
            for i in range(self.k):
                phi.append(self.extract_patch(x, l, size))
                size = int(self.s * size)

            # resize the patches to squares of size g
            for i in range(1, len(phi)):
                k = phi[i].shape[-1] // self.g
                phi[i] = functional.avg_pool2d(phi[i], k)

            # concatenate into a single tensor and flatten
            phi = torch.cat(phi, 1)
            phi = phi.view(phi.shape[0], -1)

            return phi

        def extract_patch(self, x, l, size) -> torch.Tensor:
            """Extract a single patch for each image in `x`.

            Args:
            x: a 4D Tensor of shape (B, H, W, C). The minibatch
            of images.
            l: a 2D Tensor of shape (B, 2).
            size: a scalar defining the size of the extracted patch.

            Returns:
            patch: a 4D Tensor of shape (B, size, size, C)"""
            B, C, H, W = x.shape

            start = self.denormalize(H, l)
            end = start + size

            # pad with zeros
            x = functional.pad(x, (size // 2, size // 2, size // 2, size // 2))

            # loop through mini-batch and extract patches
            patch = []
            for i in range(B):
                patch.append(x[i, :, start[i, 1] : end[i, 1], start[i, 0] : end[i, 0]])
            return torch.stack(patch)

        def denormalize(self, T, coords) -> torch.LongTensor:
            """Convert coordinates in the range [-1, 1] to
            coordinates in the range [0, T] where `T` is
            the size of the image."""
            return (0.5 * ((coords + 1.0) * T)).long()

        def exceeds(self, from_x, to_x, from_y, to_y, T) -> bool:
            """Check whether the extracted patch will exceed
            the boundaries of the image of size `T`."""
            if (from_x < 0) or (from_y < 0) or (to_x > T) or (to_y > T):
                return True
            return False

    def __init__(self, h_g, h_l, g, k, s, c):
        super().__init__()

        self.retina = self.Retina(g, k, s)

        self.fc1 = nn.Linear(
            k * g * g * c, h_g
        )  # glimpse layer TODO: RENAME TO WHAT IS IT!!

        self.fc2 = nn.Linear(2, h_l)  # location layer

        self.fc3 = nn.Linear(h_g, h_g + h_l)
        self.fc4 = nn.Linear(h_l, h_g + h_l)

    def forward(self, x: torch.Tensor, l_t_prev: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :type x:
        :param l_t_prev:
        :type l_t_prev:
        :return:
        :rtype:"""

        return functional.relu(
            self.fc3(
                functional.relu(self.fc1(self.retina.foveate(x, l_t_prev)))
            )  # what # generate glimpse phi from image x
            + self.fc4(
                functional.relu(self.fc2(l_t_prev.view(l_t_prev.size(0), -1)))
            )  # where
        )


class CoreRNN(nn.Module):
    """The core network.

    An RNN that maintains an internal state by integrating
    information extracted from the history of past observations.
    It encodes the agent's knowledge of the environment through
    a state vector `h_t` that gets updated at every time step `t`.

    Concretely, it takes the glimpse representation `g_t` as input,
    and combines it with its internal state `h_t_prev` at the previous
    time step, to produce the new internal state `h_t` at the current
    time step.

    In other words:

      `h_t = relu( fc(h_t_prev) + fc(g_t) )`

    Args:
      input_size: input size of the rnn.
      hidden_size: hidden size of the rnn.
      g_t: a 2D tensor of shape (B, hidden_size). The glimpse
          representation returned by the glimpse network for the
          current timestep `t`.
      h_t_prev: a 2D tensor of shape (B, hidden_size). The
          hidden state vector for the previous timestep `t-1`.

    Returns:
      h_t: a 2D tensor of shape (B, hidden_size). The hidden
          state vector for the current timestep `t`."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def forward(self, g_t: torch.Tensor, h_t_prev: torch.Tensor) -> torch.Tensor:
        """


        :param g_t:
        :type g_t:
        :param h_t_prev:
        :type h_t_prev:
        :return:
        :rtype:"""
        h1 = self.i2h(g_t)
        h2 = self.h2h(h_t_prev)
        h_t = functional.relu(h1 + h2)
        return h_t


class Actor(nn.Module):
    """The action network.

    Uses the internal state `h_t` of the core network to
    produce the final output classification.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a softmax to create a vector of
    output probabilities over the possible classes.

    Hence, the environment action `a_t` is drawn from a
    distribution conditioned on an affine transformation
    of the hidden state vector `h_t`, or in other words,
    the action network is simply a linear softmax classifier.

    Args:
      input_size: input size of the fc layer.
      output_size: output size of the fc layer.
      h_t: the hidden state vector of the core network
          for the current time step `t`.

    Returns:
      a_t: output probability vector over the classes."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        """


        :param h_t:
        :type h_t:
        :return:
        :rtype:"""
        return functional.log_softmax(self.fc(h_t), dim=1)


class Locator(nn.Module):
    """The location network.

    Uses the internal state `h_t` of the core network to
    produce the location coordinates `l_t` for the next
    time step.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a tanh to clamp the output between
    [-1, 1]. This produces a 2D vector of means used to
    parametrize a two-component Gaussian with a fixed
    variance from which the location coordinates `l_t`
    for the next time step are sampled.

    Hence, the location `l_t` is chosen stochastically
    from a distribution conditioned on an affine
    transformation of the hidden state vector `h_t`.

    Args:
      input_size: input size of the fc layer.
      output_size: output size of the fc layer.
      std: standard deviation of the normal distribution.
      h_t: the hidden state vector of the core network for
          the current time step `t`.

    Returns:
      mu: a 2D vector of shape (B, 2).
      l_t: a 2D vector of shape (B, 2)."""

    def __init__(self, input_size: int, output_size: int, std: float):
        super().__init__()

        self.std = std

        hidden_size = input_size // 2
        self.fc = nn.Linear(input_size, hidden_size)
        self.fc_lt = nn.Linear(hidden_size, output_size)

    def forward(self, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param h_t:
        :type h_t:
        :return:
        :rtype:"""

        mu = torch.tanh(
            self.fc_lt(functional.relu(self.fc(h_t.detach())))
        )  # compute mean

        l_t = torch.distributions.Normal(
            mu, self.std
        ).rsample()  # reparametrisation trick

        # we assume both dimensions are independent
        # 1. pdf of the joint is the product of the pdfs
        # 2. log of the product is the sum of the logs
        return (
            torch.sum(Normal(mu, self.std).log_prob(l_t.detach()), dim=1),
            torch.clamp(l_t, -1, 1),  # bound between [-1, 1]
        )


class SignalBaseline(nn.Module):
    """The baseline network.

    This network regresses the baseline in the
    reward function to reduce the variance of
    the gradient update.

    Args:
      input_size: input size of the fc layer.
      output_size: output size of the fc layer.
      h_t: the hidden state vector of the core network
          for the current time step `t`.

    Returns:
      b_t: a 2D vector of shape (B, 1). The baseline
          for the current time step `t`."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        """

        :param h_t:
        :type h_t:
        :return:
        :rtype:"""
        return self.fc(h_t.detach())
