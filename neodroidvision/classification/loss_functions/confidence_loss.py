"""
A collection of loss functions for pytorch and torchvision
"""

import copy

import torch
from torch import nn
from torch.nn import functional


class ConfidenceLoss(nn.Module):
    """
    An almost copy paste of https://github.com/aivclab/confidence_classification_loss_pytorch
    The function uses the nn.Module internal state "training" to determine when to update
        lambda. You must call .eval() and .train() yourself
     Implements
       "Learning Confidence for Out-of-Distribution Detection in Neural Networks"
       https://arxiv.org/pdf/1802.04865.pdf
     A Softmax-cross-entropy classification loss, which provides an
     additional "confidence" output, which signals whether the softmax output
     is confident.
    """

    def __init__(self, hint_budget=0.3, lmbda=0.1):
        """
        Parameters
        ---------------------
        hint_budget : float
           Refer to the paper.
        lambda : float
           Refer to the paper.
        """
        super().__init__()
        self.hint_budget = hint_budget
        self._initial_lmbda = copy.copy(lmbda)
        self.lmbda = lmbda

    def reset(self):
        """
        Set lambda to initial lambda
        """
        self.lmbda = copy.copy(self._initial_lmbda)

    def _update_lmbda(self, conf_loss):
        """
        Updates current lambda. Ensure .eval() has been called if lambda should not update
        """
        if self.training:
            self.lmbda = (
                self.lmbda / 1.01
                if conf_loss.item() < self.hint_budget
                else self.lmbda / 0.99
            )

    @classmethod
    def predict(cls, input):
        """
        Compute prediction pseudo probabilities and confidence from logits

        Parameters
        ----------
        input : torch.tensor (BxK) float where K=<number_of_classes> + 1
           Classification logits + confidence logit

        Returns
        -------
        (pred, conf) prediction softmax probabilities and confidence level

        Example
        -------
          (pred, conf) = ConfidenceLoss.predict(logit_tensor)
        """
        input_pred = input[..., :-1]
        input_conf = input[..., -1]
        pred = torch.softmax(input_pred, dim=-1)
        conf = torch.sigmoid(input_conf)
        return (pred, conf)

    def forward(self, input, target):
        """
        Compute loss

        Parameters
        ----------
        input : torch.tensor (BxK) float where K=<number_of_classes> + 1
           Classification logits + confidence logit
        target : torch.tensor (Bx1) long
           Target class

        Returns
        -------
        loss value
        """
        pred_orig, conf_orig = self.predict(input)
        target_1hot = functional.one_hot(target, pred_orig.shape[-1])

        # Clamp
        eps = 1e-12
        pred_orig = torch.clamp(pred_orig, eps, 1 - eps)
        conf_orig = torch.clamp(conf_orig, eps, 1 - eps)
        # Randomly set half of predictions to 100% confidence
        b = torch.empty_like(conf_orig).uniform_(0, 1).round()
        conf_new = conf_orig * b + 1 - b
        pred_new = pred_orig * conf_new[:, None] + target_1hot * (1 - conf_new[:, None])
        pred_new = pred_new.log()
        pred_loss = functional.nll_loss(pred_new, target)
        conf_loss = -torch.log(conf_new).mean()
        tot_loss = pred_loss + self.lmbda * conf_loss

        self._update_lmbda(conf_loss)

        return tot_loss
