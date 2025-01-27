"""Focal Loss Module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    """Loss based on: https://arxiv.org/abs/1708.02002 ."""

    def __init__(self, gamma: float = 2.0, alpha=None, size_average=True):
        """Focal Loss.

        Args:
            gamma (float): value for gamma.
            alpha (float, list, torch.Tensor):
            size_average (bool): If apply an average on the loss True else False.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, inputs, targets):
        """Loss calculation.

        Args:
            inputs (torch.Tensor): Model predictions.
            targets (torch.Tensor): Labels.

        Returns: torch.Tensor
        """
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)  # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))  # N,H*W,C => N*H*W,C
        targets = targets.view(-1, 1)

        logpt = F.log_softmax(inputs, dim=-1)
        logpt = logpt.gather(1, targets)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.size_average and self.alpha is not None:
            return loss.sum() / at.sum()

        elif self.size_average:
            return loss.mean()

        else:
            return loss.sum()
