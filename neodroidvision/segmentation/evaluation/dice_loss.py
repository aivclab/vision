import numpy
import torch
from draugr.torch_utilities.operations.enums import ReductionMethodEnum
from torch import nn

__all__ = ["dice_loss", "soft_dice_coefficient", "DiceLoss", "BCEDiceLoss"]


def soft_dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float = 1e-10,
    activation: callable = torch.sigmoid,
) -> torch.Tensor:
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch"""

    if activation:
        pred = activation(pred)

    pred = pred.reshape(-1)
    target = target.reshape(-1)

    intersection = 2.0 * (pred * target).sum() + epsilon
    union = target.sum() ** 2 + pred.sum() ** 2 + epsilon

    return intersection / union


def dice_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float = 1e-10,
    activation: callable = torch.sigmoid,
) -> torch.Tensor:
    """

    Args:
      prediction:
      target:
      epsilon:

    Returns:

    """
    return 1 - soft_dice_coefficient(
        prediction, target, epsilon=epsilon, activation=activation
    )


class DiceLoss(nn.Module):
    """ """

    def __init__(self, *, epsilon: float = 1e-10, activation: callable = torch.sigmoid):
        super().__init__()
        self.activation = activation
        self.epsilon = epsilon

    def forward(self, y_pr: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
        """

        Args:
          y_pr:
          y_gt:

        Returns:

        """
        return dice_loss(y_pr, y_gt, epsilon=self.epsilon, activation=self.activation)


class BCEDiceLoss(DiceLoss):
    """ """

    def __init__(
        self,
        *,
        epsilon: float = 1e-7,
        activation: callable = torch.sigmoid,
        lambda_dice: float = 1.0,
        lambda_bce: float = 1.0,
        reduction: ReductionMethodEnum = ReductionMethodEnum.mean,
    ):
        super().__init__(epsilon=epsilon, activation=activation)

        reduction = ReductionMethodEnum(reduction)
        if activation == None:
            self.bce = nn.BCELoss(reduction=reduction.value)
        else:
            self.bce = nn.BCEWithLogitsLoss(reduction=reduction.value)

        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce

    def forward(self, y_pr: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
        """

        Args:
          y_pr:
          y_gt:

        Returns:

        """
        return (self.lambda_dice * super().forward(y_pr, y_gt)) + (
            self.lambda_bce * self.bce(y_pr, y_gt)
        )


if __name__ == "__main__":
    numpy.random.seed(2)
    data = numpy.random.random_sample((2, 1, 84, 84))
    a = torch.FloatTensor(data)
    b = torch.FloatTensor(data.transpose((0, 1, 3, 2)))
    print(soft_dice_coefficient(a, torch.sigmoid(a)))
    print(dice_loss(a, a))
    print(dice_loss(a, b))

    h = torch.FloatTensor(numpy.array([[0, 1], [1, 1]]))
    j = torch.FloatTensor(numpy.ones((2, 2)))
    print(dice_loss(h, j))
    print(dice_loss(j, h))
