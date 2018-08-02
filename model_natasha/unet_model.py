# full assembly of the sub-parts to form the complete net

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, with_depth=False):
        super(UNet, self).__init__()
        self.inc = Inconv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        if with_depth:
            self.up1 = Up(1024 + 1, 256)
        else:
            self.up1 = Up(1024, 256)

        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = Outconv(64, n_classes)

    def forward(self, x, depth=None):
        # we need an input image which sizes are divisible by 32
        # [8, 4, 101, 101] -> [8, 64, 101, 101]
        x1 = self.inc(x)
        # [8, 64, 101, 101]-> [8, 128, 50, 50]
        x2 = self.down1(x1)
        # [8, 128, 50, 50] -> [8, 256, 25, 25]
        x3 = self.down2(x2)
        # [8, 256, 25, 25] -> [8, 512, 12, 12]
        x4 = self.down3(x3)
        # [8, 512, 12, 12] -> [8, 512, 6, 6]
        x5 = self.down4(x4)

        if depth is not None:
            depth_layer = depth.unsqueeze(dim=1)[:, :, :x5.shape[2], :x5.shape[3]]
            x5 = torch.cat((x5, depth_layer), dim=1)

        x = self.up1(x5, x4) # [8, 256, 12, 12]
        x = self.up2(x, x3) # [8, 128, 24, 24]
        x = self.up3(x, x2) # [8, 64, 48, 48]
        x = self.up4(x, x1) # [8, 64, 96, 96]
        x = self.outc(x) # [8, 2, 96, 96]

        return x
