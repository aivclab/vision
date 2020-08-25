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

    def curl():
        import torch
        from torchvision import models

        resnet = models.resnet50(pretrained=True)

        learner = ContrastiveLearner(
            resnet,
            image_size=256,
            hidden_layer="avgpool",
            use_momentum=True,  # use momentum for key encoder
            momentum_value=0.999,
            project_hidden=False,  # no projection heads
            use_bilinear=True,  # in paper, logits is bilinear product of query / key
            use_nt_xent_loss=False,  # use regular contrastive loss
            augment_both=False,  # in curl, only the key is augmented
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
            learner.update_moving_average()  # update moving average of key encoder

    curl()
