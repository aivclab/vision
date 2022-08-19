#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 5/5/22

          At any representation level, domain and representation agnostic !




           """
__all__ = []


def gauss_sample_deep_aug():
    """Use category balanced batches, ensuring at least 3-5 instance of a category. Then per category in batch, find local
    manifold by finding the
    covariance matrix at
     representation level
    and
    sample a
    multi
    var gauss
    for new
    samples
    for further forward passes"""

    pass

    class BatchAugmentation(object):
        """description"""

        def __init__(self):
            pass

        def __call__(self, batch):
            pass


def space_partition_deep_aug():
    """
    Partition the space into a grid of regions or clusters and sample from the vicinity of each region.

    :return:
    :rtype:
    """
    pass


def category_cover_space_sample_deep_aug():
    """
    cover the space with n gaussians(or other distribution) for each category and sample along ridges

    :return:
    :rtype:
    """
    pass


def linear_interpolate_pairs_deep_aug():
    """

    Find a pair and linearly interpolate between pairs of each category

    :return:
    :rtype:
    """

    pass


def non_linear_interpolate_pairs_deep_aug():
    """

    Find a pair and use a learned interpolator to interpolate between points.

    for the learned interpolation regularize the representation by adding a small amount of noise to the representation abd impose penalty on l2 norms, and so on to avoid degenaracy/mode collapse. Same a avoid overfitting a generative model.

    Look at variational autoencoders for this.


    OTHER NOTES:
    Ties to generative diffusion models?

    Look up LINDA: learning to interpolate for data augmentation.
    and SSMBA: state space model for data augmentation.
    and Mixup: non-local data augmentation.

    Domain specific: combine multiple Facial features from different people. set of eye from person A  and mouth from person B.

    :return:
    :rtype:
    """

    pass
