import numpy
import torch

from neodroidvision.regression.generative.vae.vqvae2.vq import embedding_distances


def test_embedding_distances():
  dictionary = torch.randn(15, 7)
  tensor = torch.randn(25, 13, 7)
  with torch.no_grad():
    actual = embedding_distances(dictionary, tensor).numpy()
    expected = naive_embedding_distances(dictionary, tensor).numpy()
    assert numpy.allclose(actual, expected, atol=1e-4)


def naive_embedding_distances(dictionary, tensor):
  return torch.sum(torch.pow(tensor[..., None, :] - dictionary, 2), dim=-1)
