#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from samples.reconstruction.vanilla_vae.data_loader import load_binary_mnist
from samples.reconstruction.vanilla_vae.models import Generator, VariationalFlow, VariationalMeanField
from warg.named_ordered_dictionary import NOD

__author__ = 'cnheider'
__doc__ = r'''
Fit a variational autoencoder to MNIST. 
           '''

import torch.utils.data
import numpy as np
import random

from neodroidvision import PROJECT_APP_PATH


def evaluate(generator,
             evaluation_data,
             ddevice):
  generator.eval()
  for batch in evaluation_data:
    x = batch[0].to(ddevice)
    z = torch.randn(cfg.latent_size, device=ddevice)
    log_p_x_and_z, logits = generator(z, x)
    print(log_p_x_and_z)


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

  evaluate(
      generator,

      test_data,
      next(generator.parameters()).device)
