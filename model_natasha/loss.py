import sys
import torch
from os import path
import torch.nn.functional as F
from torch.autograd import Function


def bce(y_input, y_target):
    loss = torch.nn.BCELoss()
    return loss(y_input, y_target)