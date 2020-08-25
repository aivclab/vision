#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 07/07/2020
           """

from neodroidvision.classification.architectures.contrastive.contrastive_learner import (
    ContrastiveLearner,
)

if __name__ == "__main__":

    def simclr():
        import torch
        from torchvision import models

        resnet = models.resnet50(pretrained=True)

        learner = ContrastiveLearner(
            resnet,
            image_size=256,
            hidden_layer="avgpool",  # layer name where output is hidden dimension. this can also be an integer specifying the index of the child
            project_hidden=True,  # use projection head
            project_dim=128,  # projection head dimensions, 128 from paper
            use_nt_xent_loss=True,  # the above mentioned loss, abbreviated
            temperature=0.1,  # temperature
            augment_both=True,  # augment both query and key
        )

        opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

        def sample_batch_images():
            return torch.randn(20, 3, 256, 256)

        for _ in range(100):
            images = sample_batch_images()
            loss = learner(images)
            opt.zero_grad()
            loss.backward()
            opt.step()

    simclr()
