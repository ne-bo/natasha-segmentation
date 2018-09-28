import sys
import torch
from os import path
import torch.nn.functional as F
from torch.autograd import Function

from model_natasha.lovasz_losses import lovasz_hinge


def bce(y_input, y_target):
    loss = torch.nn.BCELoss()
    return loss(y_input, y_target)


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=0, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, output, target):
        return 1 - (2 * torch.sum(output * target) + self.smooth) / (
                torch.sum(output) + torch.sum(target) + self.smooth + self.eps)


def dice(y_input, y_target):
    loss = DiceLoss()
    return loss(y_input, y_target)


def lovasz(y_input, y_target):
    return lovasz_hinge(y_input, y_target, per_image=False)


def lovasz_bce(y_input, y_target):
    loss = torch.nn.BCELoss()
    return loss(y_input, y_target) + lovasz_hinge(y_input, y_target, per_image=False)


def dice_bce(y_input, y_target):
    bce = torch.nn.BCELoss()
    dice = DiceLoss()
    return dice(y_input, y_target) + bce(y_input, y_target)
