#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
           """

from typing import Tuple

import numpy
import torch
import torch.utils
import torch.utils.data
from draugr.torch_utilities.operations.enums import ReductionMethodEnum
from torch import nn

from .vae_flow import FlowSequential, InverseAutoregressiveFlow, Reverse


class MLP(nn.Module):
    """ """

    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        modules = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        ]
        self.net = nn.Sequential(*modules)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """

        :param input:
        :type input:
        :return:
        :rtype:"""
        return self.net(input)


class Generator(nn.Module):
    """
    Bernoulli model parameterized by a generative network with Gaussian latents for MNIST."""

    def __init__(self, latent_size, data_size):
        super().__init__()
        self.register_buffer("p_z_loc", torch.zeros(latent_size))
        self.register_buffer("p_z_scale", torch.ones(latent_size))
        self.log_p_z = NormalLogProb()
        self.log_p_x = BernoulliLogProb()
        self.generative_network = MLP(
            input_size=latent_size, output_size=data_size, hidden_size=latent_size * 2
        )

    def forward(self, z, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return log probability of model."""
        log_p_z = self.log_p_z(self.p_z_loc, self.p_z_scale, z).sum(-1, keepdim=True)
        logits = self.generative_network(z)
        # unsqueeze sample dimension
        logits, x = torch.broadcast_tensors(logits, x.unsqueeze(1))
        log_p_x = self.log_p_x(logits, x).sum(-1, keepdim=True)
        return log_p_z + log_p_x, logits


class VariationalMeanField(nn.Module):
    """
    Approximate posterior parameterized by an inference network."""

    def __init__(self, latent_size, data_size):
        super().__init__()
        self.inference_network = MLP(
            input_size=data_size,
            output_size=latent_size * 2,
            hidden_size=latent_size * 2,
        )
        self.log_q_z = NormalLogProb()
        self.softplus = nn.Softplus()

    def forward(self, x, n_samples=1):
        """
        Return sample of latent variable and log prob."""
        loc, scale_arg = torch.chunk(
            self.inference_network(x).unsqueeze(1), chunks=2, dim=-1
        )
        scale = self.softplus(scale_arg)
        eps = torch.randn((loc.shape[0], n_samples, loc.shape[-1]), device=loc.device)
        z = loc + scale * eps  # reparameterization
        log_q_z = self.log_q_z(loc, scale, z).sum(-1, keepdim=True)
        return z, log_q_z


class VariationalFlow(nn.Module):
    """
    Approximate posterior parameterized by a flow (https://arxiv.org/abs/1606.04934)."""

    def __init__(self, latent_size, data_size, flow_depth):
        super().__init__()
        hidden_size = latent_size * 2
        self.inference_network = MLP(
            input_size=data_size,
            # loc, scale, and context
            output_size=latent_size * 3,
            hidden_size=hidden_size,
        )
        modules = []

        for _ in range(flow_depth):
            modules.append(
                InverseAutoregressiveFlow(
                    num_input=latent_size,
                    num_hidden=hidden_size,
                    num_context=latent_size,
                )
            )
            modules.append(Reverse(latent_size))
        self.q_z_flow = FlowSequential(*modules)

        self.log_q_z_0 = NormalLogProb()
        self.softplus = nn.Softplus()

    def forward(self, x, n_samples=1):
        """
        Return sample of latent variable and log prob."""
        loc, scale_arg, h = torch.chunk(
            self.inference_network(x).unsqueeze(1), chunks=3, dim=-1
        )
        scale = self.softplus(scale_arg)
        eps = torch.randn((loc.shape[0], n_samples, loc.shape[-1]), device=loc.device)
        z_0 = loc + scale * eps  # reparameterization
        log_q_z_0 = self.log_q_z_0(loc, scale, z_0)
        z_T, log_q_z_flow = self.q_z_flow(z_0, context=h)
        log_q_z = (log_q_z_0 + log_q_z_flow).sum(-1, keepdim=True)
        return z_T, log_q_z


class NormalLogProb(nn.Module):
    """ """

    def forward(self, loc, scale, z):
        """

        :param loc:
        :type loc:
        :param scale:
        :type scale:
        :param z:
        :type z:
        :return:
        :rtype:"""
        var = scale**2
        return -0.5 * torch.log(2 * numpy.pi * var) - torch.pow(z - loc, 2) / (2 * var)


class BernoulliLogProb(nn.Module):
    """ """

    def __init__(self):
        super().__init__()
        self.bce_with_logits = nn.BCEWithLogitsLoss(
            reduction=ReductionMethodEnum.none.value
        )

    def forward(self, logits, target):
        """bernoulli log prob is equivalent to negative binary cross entropy"""
        return -self.bce_with_logits(logits, target)
