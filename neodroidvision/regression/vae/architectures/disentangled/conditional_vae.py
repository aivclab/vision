from typing import Sequence, Tuple

import torch
from torch import nn

from neodroidvision.regression.vae.architectures.vae import VAE

__all__ = ["ConditionalVAE"]


class Encoder(nn.Module):
    """ """

    def __init__(
        self, layer_sizes: Sequence[int], latent_size: int, num_conditions: int
    ):
        super().__init__()

        self.input_size = layer_sizes[0]
        self.multi_layer_perceptron = nn.Sequential()
        layer_sizes[0] += num_conditions

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.multi_layer_perceptron.add_module(
                name=f"L{i:d}", module=nn.Linear(in_size, out_size)
            )
            self.multi_layer_perceptron.add_module(name=f"A{i:d}", module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
          x:
          condition:

        Returns:

        """
        x = torch.cat((x.reshape(-1, self.input_size), condition), dim=-1)

        x = self.multi_layer_perceptron(x)

        mean = self.linear_means(x)
        log_var = self.linear_log_var(x)

        return mean, log_var


class Decoder(nn.Module):
    """ """

    def __init__(
        self, layer_sizes: Sequence[int], latent_size: int, num_conditions: int
    ):

        super().__init__()

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(
            zip([latent_size + num_conditions] + layer_sizes[:-1], layer_sizes)
        ):
            self.MLP.add_module(name=f"L{i:d}", module=nn.Linear(in_size, out_size))
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(name=f"A{i:d}", module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """

        Args:
          z:
          condition:

        Returns:

        """
        z_cat = torch.cat((z, condition), dim=-1)
        x = self.MLP(z_cat)
        return x.view(-1, 28, 28)


class ConditionalVAE(VAE):
    """ """

    def encode(self, *x: torch.Tensor) -> torch.Tensor:
        """

        Args:
          *x:

        Returns:

        """
        return self.encoder(*x)

    def decode(self, *x: torch.Tensor) -> torch.Tensor:
        """

        Args:
          *x:

        Returns:

        """
        return self.decoder(*x)

    def __init__(
        self,
        encoder_layer_sizes: Sequence[int],
        latent_size: int,
        num_conditions: int,
        *,
        decoder_layer_sizes: Sequence[int] = None,
    ):
        super().__init__(latent_size)

        assert num_conditions > 1

        assert isinstance(encoder_layer_sizes, Sequence)
        assert isinstance(latent_size, int)
        if decoder_layer_sizes:
            assert isinstance(decoder_layer_sizes, Sequence)
        else:
            raise NotImplementedError  # TODO infer / reversed encoder

        self.encoder = Encoder(encoder_layer_sizes, latent_size, num_conditions)
        self.decoder = Decoder(decoder_layer_sizes, latent_size, num_conditions)

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Args:
          x:
          condition:

        Returns:

        """
        mean, log_var = self.encode(x, condition)
        z = self.reparameterise(mean, log_var)
        return self.decode(z, condition), mean, log_var, z
