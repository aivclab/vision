import torch
from torch import nn
from torch.autograd.function import Function

__all__ = ["CenterLoss", "CenterLossFunc"]


class CenterLoss(nn.Module):
    """ """

    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(
            torch.randn(num_classes, feat_dim), requires_grad=True
        )
        self.center_loss_func = CenterLossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, label: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """

        :param label:
        :type label:
        :param feat:
        :type feat:
        :return:
        :rtype:"""
        batch_size = feat.size(0)
        feat = feat.reshape(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError(
                f"Center's dim: {self.feat_dim} should be equal to input feature's \
    dim: {feat.size(1)}"
            )
        batch_size_tensor = feat.new_empty(1).fill_(
            batch_size if self.size_average else 1
        )
        loss = self.center_loss_func(feat, label, self.centers, batch_size_tensor)
        return loss


class CenterLossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size) -> torch.Tensor:
        """

        :param ctx:
        :type ctx:
        :param feature:
        :type feature:
        :param label:
        :type label:
        :param centers:
        :type centers:
        :param batch_size:
        :type batch_size:
        :return:
        :rtype:"""
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        """

        :param ctx:
        :type ctx:
        :param grad_output:
        :type grad_output:
        :return:
        :rtype:"""
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(
            0, label.unsqueeze(1).expand(feature.size()).long(), diff
        )
        grad_centers = grad_centers / counts.reshape(-1, 1)
        return -grad_output * diff / batch_size, None, grad_centers / batch_size, None


if __name__ == "__main__":

    def main():
        """ """
        from draugr.torch_utilities import global_torch_device

        torch.manual_seed(999)
        print("-" * 80)
        ct = CenterLoss(10, 2, size_average=True).to(global_torch_device())
        y = torch.Tensor([0, 0, 2, 1]).to(global_torch_device())
        feat = torch.zeros(4, 2).to(global_torch_device()).requires_grad_()
        print(list(ct.parameters()))
        print(ct.centers.grad)
        out = ct(y, feat)
        print(out.item())
        out.backward()
        print(ct.centers.grad)
        print(feat.grad)

    main()
