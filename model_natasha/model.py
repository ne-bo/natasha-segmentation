import sys

import torch

from base import BaseModel
from os import path

from model_natasha.unet_model import UNet

# from pretrainedmodels import


class NatashaSegmentation(BaseModel):
    def __init__(self, config):
        super(NatashaSegmentation, self).__init__(config)
        self.config = config
        self.net = UNet(n_channels=3, n_classes=1, with_depth=config['with_depth']).cuda()

    def forward(self, x, depth=None):
        x = self.net(x, depth)
        return x

