import torch
import torch.nn as nn

from neodroidvision.reconstruction.generative.vae.architectures.vae import VAE


class ConditionalVAE(VAE):

  def encode(self, *x: torch.Tensor) -> torch.Tensor:
    return self.encoder(*x)

  def decode(self, *x: torch.Tensor) -> torch.Tensor:
    return self.decoder(*x)

  def __init__(self,
               encoder_layer_sizes,
               latent_size,
               decoder_layer_sizes,
               num_conditions):
    super().__init__(latent_size)

    assert num_conditions > 1

    assert type(encoder_layer_sizes) == list
    assert type(latent_size) == int
    assert type(decoder_layer_sizes) == list

    self.encoder = Encoder(encoder_layer_sizes,
                           latent_size,
                           num_conditions
                           )

    self.decoder = Decoder(decoder_layer_sizes,
                           latent_size,
                           num_conditions
                           )

  def forward(self, x: torch.Tensor, condition: torch.Tensor):
    mean, log_var = self.encode(x, condition)

    z = self.reparameterise(mean, log_var)
    reconstruction = self.decode(z, condition)

    return reconstruction, mean, log_var, z


class Encoder(nn.Module):

  def __init__(self,
               layer_sizes,
               latent_size,
               num_conditions
               ):

    super().__init__()

    self.input_size = layer_sizes[0]
    self.MLP = nn.Sequential()
    layer_sizes[0] += num_conditions

    for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
      self.MLP.add_module(name=f"L{i:d}", module=nn.Linear(in_size, out_size))
      self.MLP.add_module(name=f"A{i:d}", module=nn.ReLU())

    self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
    self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

  def forward(self, x, condition):
    x = torch.cat((x.view(-1, self.input_size), condition), dim=-1)

    x = self.MLP(x)

    mean = self.linear_means(x)
    log_var = self.linear_log_var(x)

    return mean, log_var


class Decoder(nn.Module):

  def __init__(self,
               layer_sizes,
               latent_size,
               num_conditions
               ):

    super().__init__()

    self.MLP = nn.Sequential()

    for i, (in_size, out_size) in enumerate(zip([latent_size + num_conditions] + layer_sizes[:-1],
                                                layer_sizes)):
      self.MLP.add_module(name=f"L{i:d}", module=nn.Linear(in_size, out_size))
      if i + 1 < len(layer_sizes):
        self.MLP.add_module(name=f"A{i:d}", module=nn.ReLU())
      else:
        self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

  def forward(self, z, condition):
    z_cat = torch.cat((z, condition), dim=-1)
    x = self.MLP(z_cat)
    return x.view(-1, 28, 28)
