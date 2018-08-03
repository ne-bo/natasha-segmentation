import sys

import torch

from base import BaseModel
from os import path

from model_natasha.unet_model import UNet, UNetVGG16, UNet11, UNetResNet


from pretrainedmodels import se_resnet50


class NatashaSegmentation(BaseModel):
    def __init__(self, config):
        super(NatashaSegmentation, self).__init__(config)
        self.config = config
        # self.net = UNet(n_channels=3, n_classes=1, with_depth=config['with_depth']).cuda()
        self.net = UNetVGG16(num_classes=1, num_filters=32,
                             dropout_2d=0.2, pretrained=True, is_deconv=False, with_depth=config['with_depth']).cuda()
        # self.net = UNet11(num_classes=1, num_filters=32, pretrained=True, with_depth=config['with_depth']).cuda()
        # self.net = UNetResNet(encoder_depth=34, num_classes=1, num_filters=32, dropout_2d=0.2,
        #                      pretrained=True, is_deconv=False, with_depth=config['with_depth']).cuda()

    def forward(self, x, depth=None):
        x = self.net(x, depth)
        return x
