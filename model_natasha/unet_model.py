# full assembly of the sub-parts to form the complete net
import pretrainedmodels
import torchvision

from .unet_parts import *


class CSE(nn.Module):
    def __init__(self, in_ch, r):
        super(CSE, self).__init__()

        self.linear_1 = nn.Linear(in_ch, in_ch // r)
        self.linear_2 = nn.Linear(in_ch // r, in_ch)

    def forward(self, x):
        input_x = x

        x = x.view(*(x.shape[:-2]), -1).mean(-1)
        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.sigmoid(x)
        x = input_x * x

        return x


class SSE(nn.Module):
    def __init__(self, in_ch):
        super(SSE, self).__init__()

        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1)

    def forward(self, x):
        input_x = x

        x = self.conv(x)
        x = F.sigmoid(x)

        x = input_x * x

        return x


class SCSE(nn.Module):
    def __init__(self, in_ch, r):
        super(SCSE, self).__init__()

        self.cSE = CSE(in_ch, r)
        self.sSE = SSE(in_ch)

    def forward(self, x):
        cSE = self.cSE(x)
        sSE = self.sSE(x)

        x = cSE + sSE

        return x


class SqEx(nn.Module):

    def __init__(self, n_features, reduction=16):
        super(SqEx, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):
        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True)
                                  )

    def forward(self, x):
        return self.conv(x)


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

        x = self.up1(x5, x4)  # [8, 256, 12, 12]
        x = self.up2(x, x3)  # [8, 128, 24, 24]
        x = self.up3(x, x2)  # [8, 64, 48, 48]
        x = self.up4(x, x1)  # [8, 64, 96, 96]
        x = self.outc(x)  # [8, 2, 96, 96]

        return x


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class UNetVGG16(nn.Module):
    """PyTorch U-Net model using VGG16 encoder.

    UNet: https://arxiv.org/abs/1505.04597
    VGG: https://arxiv.org/abs/1409.1556
    Proposed by Vladimir Iglovikov and Alexey Shvets: https://github.com/ternaus/TernausNet

    Args:
            num_classes (int): Number of output classes.
            num_filters (int, optional): Number of filters in the last layer of decoder. Defaults to 32.
            dropout_2d (float, optional): Probability factor of dropout layer before output layer. Defaults to 0.2.
            pretrained (bool, optional):
                False - no pre-trained weights are being used.
                True  - VGG encoder is pre-trained on ImageNet.
                Defaults to False.
            is_deconv (bool, optional):
                False: bilinear interpolation is used in decoder.
                True: deconvolution is used in decoder.
                Defaults to False.

    """

    def __init__(self, num_classes=1, num_filters=32, dropout_2d=0.2,
                 pretrained=False, is_deconv=False, with_depth=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder[0],
                                   nn.BatchNorm2d(num_features=self.encoder[0].out_channels),
                                   self.relu,
                                   self.encoder[2],
                                   nn.BatchNorm2d(num_features=self.encoder[2].out_channels),
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   nn.BatchNorm2d(num_features=self.encoder[5].out_channels),
                                   self.relu,
                                   self.encoder[7],
                                   nn.BatchNorm2d(num_features=self.encoder[7].out_channels),
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   nn.BatchNorm2d(num_features=self.encoder[10].out_channels),
                                   self.relu,
                                   self.encoder[12],
                                   nn.BatchNorm2d(num_features=self.encoder[12].out_channels),
                                   self.relu,
                                   self.encoder[14],
                                   nn.BatchNorm2d(num_features=self.encoder[14].out_channels),
                                   self.relu)
        self.conv4 = nn.Sequential(self.encoder[17],
                                   nn.BatchNorm2d(num_features=self.encoder[17].out_channels),
                                   self.relu,
                                   self.encoder[19],
                                   nn.BatchNorm2d(num_features=self.encoder[19].out_channels),
                                   self.relu,
                                   self.encoder[21],
                                   nn.BatchNorm2d(num_features=self.encoder[21].out_channels),
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   nn.BatchNorm2d(num_features=self.encoder[24].out_channels),
                                   self.relu,
                                   self.encoder[26],
                                   nn.BatchNorm2d(num_features=self.encoder[26].out_channels),
                                   self.relu,
                                   self.encoder[28],
                                   nn.BatchNorm2d(num_features=self.encoder[28].out_channels),
                                   self.relu)

        if with_depth:
            self.center = DecoderBlockV2(512 + 1, num_filters * 8 * 2, num_filters * 8, is_deconv)
            self.dec5 = DecoderBlockV2(512 + num_filters * 8 + 1, num_filters * 8 * 2, num_filters * 8, is_deconv)
        else:
            self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)
            self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec4 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x, depth=None):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        if depth is not None:
            depth_layer = depth.unsqueeze(dim=1)[:, :, :conv5.shape[2], :conv5.shape[3]]
            conv5 = torch.cat((conv5, depth_layer), dim=1)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        return self.final(F.dropout2d(dec1, p=self.dropout_2d))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet11(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False, with_depth=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with VGG11
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.vgg11(pretrained=pretrained).features

        self.relu = self.encoder[1]
        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]

        if with_depth:
            self.center = DecoderBlock(num_filters * 8 * 2 + 1, num_filters * 8 * 2, num_filters * 8)
            self.dec5 = DecoderBlock(num_filters * (16 + 8) + 1, num_filters * 8 * 2, num_filters * 8)
        else:
            self.center = DecoderBlock(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)
            self.dec5 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)

        self.dec4 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.dec3 = DecoderBlock(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x, depth=None):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s))
        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))

        if depth is not None:
            depth_layer = depth.unsqueeze(dim=1)[:, :, :conv5.shape[2], :conv5.shape[3]]
            conv5 = torch.cat((conv5, depth_layer), dim=1)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)


class UNetResNet(nn.Module):
    """PyTorch U-Net model using ResNet(34, 101 or 152) encoder.

    UNet: https://arxiv.org/abs/1505.04597
    ResNet: https://arxiv.org/abs/1512.03385
    Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/

    Args:
            encoder_depth (int): Depth of a ResNet encoder (34, 101 or 152).
            num_classes (int): Number of output classes.
            num_filters (int, optional): Number of filters in the last layer of decoder. Defaults to 32.
            dropout_2d (float, optional): Probability factor of dropout layer before output layer. Defaults to 0.2.
            pretrained (bool, optional):
                False - no pre-trained weights are being used.
                True  - ResNet encoder is pre-trained on ImageNet.
                Defaults to False.
            is_deconv (bool, optional):
                False: bilinear interpolation is used in decoder.
                True: deconvolution is used in decoder.
                Defaults to False.

    """

    def __init__(self, encoder_depth, num_classes, num_filters=32, dropout_2d=0.2,
                 pretrained=False, is_deconv=False, with_depth=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        # self.sqex1 = SCSE(in_ch=64, r=16)
        # self.sqex2 = SCSE(in_ch=256, r=16)
        # self.sqex3 = SCSE(in_ch=512, r=16)
        # self.sqex4 = SCSE(in_ch=1024, r=16)
        # self.sqex5 = SCSE(in_ch=2048, r=16)
        #
        # self.sqex_d5 = SCSE(in_ch=256, r=16)
        # self.sqex_d4 = SCSE(in_ch=256, r=16)
        # self.sqex_d3 = SCSE(in_ch=64, r=16)
        # self.sqex_d2 = SCSE(in_ch=128, r=16)
        # self.sqex_d1 = SCSE(in_ch=32, r=16)
        # self.sqex_d0 = SCSE(in_ch=32, r=16)

        if encoder_depth == 34:
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 101:
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            # self.encoder = torchvision.models.resnet152(pretrained=pretrained)
            self.encoder = pretrainedmodels.se_resnet152(pretrained='imagenet')
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')

        print(self.encoder)

        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        # self.conv1 = nn.Sequential(self.encoder.layer0.conv1,
        #                            self.encoder.layer0.bn1,
        #                            self.encoder.layer0.relu1,
        #                            self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        if with_depth:
            self.center = DecoderBlockV2(bottom_channel_nr + 1, num_filters * 8 * 2, num_filters * 8, is_deconv)
            self.dec5 = DecoderBlockV2(bottom_channel_nr + num_filters * 8 + 1, num_filters * 8 * 2, num_filters * 8,
                                       is_deconv)
        else:
            self.center = DecoderBlockV2(bottom_channel_nr, num_filters * 8 * 2, num_filters * 8, is_deconv)
            self.dec5 = DecoderBlockV2(bottom_channel_nr + num_filters * 8, num_filters * 8 * 2, num_filters * 8,
                                       is_deconv)

        self.dec4 = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8,
                                   is_deconv)
        self.dec3 = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2,
                                   is_deconv)
        self.dec2 = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2,
                                   is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x, depth=None):
        conv1 = self.conv1(x)  # self.sqex1()
        conv2 = self.conv2(conv1)  # self.sqex2()
        conv3 = self.conv3(conv2)  # self.sqex3()
        conv4 = self.conv4(conv3)  # self.sqex4()
        conv5 = self.conv5(conv4)  # self.sqex5()

        # x
        # torch.Size([48, 3, 128, 128])
        # conv1
        # torch.Size([48, 64, 32, 32])
        # conv2
        # torch.Size([48, 256, 32, 32])
        # conv3
        # torch.Size([48, 512, 16, 16])
        # conv4
        # torch.Size([48, 1024, 8, 8])
        # conv5
        # torch.Size([48, 2048, 4, 4])

        if depth is not None:
            depth_layer = depth.unsqueeze(dim=1)[:, :, :conv5.shape[2], :conv5.shape[3]]
            conv5 = torch.cat((conv5, depth_layer), dim=1)

        pool = self.pool(conv5)

        center = self.center(pool)

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        # dec5 = self.sqex_d5(self.dec5(torch.cat([center, conv5], 1)) )
        # dec4 = self.sqex_d4(self.dec4(torch.cat([dec5, conv4], 1))   )
        # dec3 = self.sqex_d3(self.dec3(torch.cat([dec4, conv3], 1))   )
        # dec2 = self.sqex_d2(self.dec2(torch.cat([dec3, conv2], 1))   )
        # dec1 = self.sqex_d1(self.dec1(dec2)                          )
        # dec0 = self.sqex_d0(self.dec0(dec1)                          )

        # dec5
        # torch.Size([48, 256, 8, 8])
        # dec4
        # torch.Size([48, 256, 16, 16])
        # dec3
        # torch.Size([48, 64, 32, 32])
        # dec2
        # torch.Size([48, 128, 64, 64])
        # dec1
        # torch.Size([48, 32, 128, 128])
        # dec0
        # torch.Size([48, 32, 128, 128])

        return self.final(F.dropout2d(dec0, p=self.dropout_2d))


class SaltUNet(nn.Module):
    def __init__(self, num_classes, dropout_2d=0.2, pretrained=False, is_deconv=False,
                 with_depth=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        self.encoder = torchvision.models.resnet34(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)

        self.input_adjust = nn.Sequential(self.encoder.conv1,
                                          self.encoder.bn1,
                                          self.encoder.relu)

        self.conv1 = list(self.encoder.layer1.children())[1]
        self.conv2 = list(self.encoder.layer1.children())[2]
        self.conv3 = list(self.encoder.layer2.children())[0]
        self.conv4 = list(self.encoder.layer2.children())[1]

        if with_depth:
            self.dec3 = DecoderBlockV2(256, 512, 256, is_deconv)
            self.dec3 = DecoderBlockV2(256 + 1, 512, 256, is_deconv)
        else:
            self.dec3 = DecoderBlockV2(256, 512, 256, is_deconv)

        self.dec2 = ConvBnRelu(256 + 64, 256)
        self.dec1 = DecoderBlockV2(256 + 64, (256 + 64) * 2, 256, is_deconv)

        self.final = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x, depth=None):
        input_adjust = self.input_adjust(x)
        conv1 = self.conv1(input_adjust)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        center = self.conv4(conv3)

        if depth is not None:
            depth_layer = depth.unsqueeze(dim=1)[:, :, :conv3.shape[2], :conv3.shape[3]]
            conv3 = torch.cat((conv3, depth_layer), dim=1)

        dec3 = self.dec3(torch.cat([center, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = F.dropout2d(self.dec1(torch.cat([dec2, conv1], 1)), p=self.dropout_2d)
        return self.final(dec1)
