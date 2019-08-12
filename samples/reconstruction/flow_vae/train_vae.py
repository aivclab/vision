#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from samples.reconstruction.vanilla_vae.data_loader import cycle, load_binary_mnist
from samples.reconstruction.vanilla_vae.models import Generator, VariationalFlow, VariationalMeanField
from warg.named_ordered_dictionary import NOD

__author__ = 'cnheider'
__doc__ = r'''
Fit a variational autoencoder to MNIST. 
           '''

import torch
import torch.utils
import torch.utils.data
import numpy as np
import random

from neodroidvision import PROJECT_APP_PATH


def evaluate(num_samples,
             generator,
             variational_encoder,
             evaluation_data,
             ddevice):
  generator.eval()
  total_log_p_x = 0.0
  total_elbo = 0.0
  for batch in evaluation_data:
    x = batch[0].to(ddevice)
    z, log_q_z = variational_encoder(x, num_samples)
    log_p_x_and_z, _ = generator(z, x)
    elbo = log_p_x_and_z - log_q_z  # importance sampling of approximate marginal likelihood with q(z) as
    # the proposal, and logsumexp in  the sample dimension
    log_p_x = torch.logsumexp(elbo, dim=1) - np.log(num_samples)
    total_elbo += elbo.cpu().numpy().mean(1).sum()  # average over sample dimension, sum over minibatch
    total_log_p_x += log_p_x.cpu().numpy().sum()  # sum over minibatch
  n_data = len(evaluation_data.dataset)
  return total_elbo / n_data, total_log_p_x / n_data


if __name__ == '__main__':

  TRAIN_DIR = (PROJECT_APP_PATH.user_data /
               'vanilla_vae' / 'train')

  if not TRAIN_DIR.exists():
    TRAIN_DIR.mkdir(parents=True)

  DATA_DIR = (PROJECT_APP_PATH.user_data /
              'vanilla_vae' / 'data')

  if not DATA_DIR.exists():
    DATA_DIR.mkdir(parents=True)

  cfg = NOD(latent_size=128,
            variational='flow',
            flow_depth=2,
            data_size=784,
            learning_rate=0.001,
            batch_size=128,
            test_batch_size=512,
            max_iterations=100000,
            log_interval=10000,
            early_stopping_interval=5,
            n_samples=128,
            use_gpu=True,
            train_dir=TRAIN_DIR,
            data_dir=DATA_DIR,
            seed=42
            )

  device = torch.device("cuda" if cfg.use_gpu else "cpu")

  torch.manual_seed(cfg.seed)
  np.random.seed(cfg.seed)
  random.seed(cfg.seed)

  generator = Generator(latent_size=cfg.latent_size,
                        data_size=cfg.data_size)

  if cfg.variational == 'flow':
    variational_encoder = VariationalFlow(latent_size=cfg.latent_size,
                                          data_size=cfg.data_size,
                                          flow_depth=cfg.flow_depth)
  elif cfg.variational == 'mean-field':
    variational_encoder = VariationalMeanField(latent_size=cfg.latent_size,
                                               data_size=cfg.data_size)
  else:
    raise ValueError('Variational distribution not implemented: %s' % cfg.variational)

  if (cfg.train_dir / 'best_state_dict').exists():
    checkpoint = torch.load(cfg.train_dir / 'best_state_dict')
    generator.load_state_dict(checkpoint['model'])
    variational_encoder.load_state_dict(checkpoint['variational'])

  generator.to(device)
  variational_encoder.to(device)

  parameters = list(generator.parameters()) + list(variational_encoder.parameters())
  optimizer = torch.optim.RMSprop(parameters,
                                  lr=cfg.learning_rate,
                                  centered=True)

  kwargs = {'num_workers':4, 'pin_memory':True} if cfg.use_gpu else {}
  train_data, valid_data, test_data = load_binary_mnist(cfg, **kwargs)

  best_valid_elbo = -np.inf
  num_no_improvement = 0

  for step, batch in enumerate(cycle(train_data)):
    x = batch[0].to(device)
    generator.zero_grad()
    variational_encoder.zero_grad()
    z, log_q_z = variational_encoder(x, n_samples=1)
    log_p_x_and_z, _ = generator(z, x)
    elbo = (log_p_x_and_z - log_q_z).mean(1)  # average over sample dimension
    loss = -elbo.sum(0)  # sum over batch dimension
    loss.backward()
    optimizer.step()

    if step % cfg.log_interval == 0:
      print(f'step:\t{step}\ttrain elbo: {elbo.detach().cpu().numpy().mean():.2f}')
      with torch.no_grad():
        valid_elbo, valid_log_p_x = evaluate(cfg.n_samples,
                                             generator,
                                             variational_encoder,
                                             valid_data,
                                             next(generator.parameters()).device)
      print(f'step:\t{step}\t\tvalid elbo: {valid_elbo:.2f}\tvalid log p(x): {valid_log_p_x:.2f}')
      if valid_elbo > best_valid_elbo:
        num_no_improvement = 0
        best_valid_elbo = valid_elbo
        states = {'model':      generator.state_dict(),
                  'variational':variational_encoder.state_dict()
                  }
        torch.save(states, cfg.train_dir / 'best_state_dict')
      else:
        num_no_improvement += 1

      if num_no_improvement > cfg.early_stopping_interval:
        checkpoint = torch.load(cfg.train_dir / 'best_state_dict')
        generator.load_state_dict(checkpoint['model'])
        variational_encoder.load_state_dict(checkpoint['variational'])
        with torch.no_grad():
          test_elbo, test_log_p_x = evaluate(cfg.n_samples,
                                             generator,
                                             variational_encoder,
                                             test_data,
                                             next(generator.parameters()).device)
        print(f'step:\t{step}\t\ttest elbo: {test_elbo:.2f}\ttest log p(x): {test_log_p_x:.2f}')
        break
