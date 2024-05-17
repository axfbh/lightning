import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial
from torchvision.ops.misc import Conv2dNormActivation
from ops.models.neck.spp import SPP

BN = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
CBM = partial(Conv2dNormActivation, bias=False, inplace=True, norm_layer=BN, activation_layer=nn.Mish)


class ResidualLayer(nn.Module):
    def __init__(self, in_ch, out_ch, shortcut=True):
        super(ResidualLayer, self).__init__()
        self.shortcut = shortcut
        self.conv = nn.Sequential(
            CBM(in_ch, out_ch, 1),
            CBM(out_ch, in_ch, 3),
        )

    def forward(self, x):
        return x + self.conv(x) if self.shortcut else self.conv(x)


class WrapLayer(nn.Module):
    def __init__(self, c1, c2, count=1, shortcut=True, first=False):
        super(WrapLayer, self).__init__()
        c_ = c1 if first else c1 // 2
        self.trans_0 = CBM(c1, c_, 1)

        self.trans_1 = CBM(c1, c_, 1)

        self.make_layers = nn.ModuleList()
        for _ in range(count):
            self.make_layers.append(ResidualLayer(c_, c_, shortcut))

        self.conv1 = CBM(c_, c_, 1)

        self.trans_cat = CBM(c1 * 2 if first else c1, c2, 1)

    def forward(self, x):
        # ----------- 两分支 -----------
        out0 = self.trans_0(x)
        out1 = self.trans_1(x)

        for conv in self.make_layers:
            out0 = conv(out0)

        out0 = self.conv1(out0)
        out = torch.cat([out0, out1], 1)
        out = self.trans_cat(out)
        return out


class CSPDarknetV1(nn.Module):
    def __init__(self, base_channels, base_depth, num_classes=100):
        super(CSPDarknetV1, self).__init__()

        DownSampleLayer = partial(CBM, kernel_size=3, stride=2)

        self.stem = nn.Sequential(
            CBM(3, base_channels, 3),
            DownSampleLayer(base_channels, base_channels * 2),
            WrapLayer(base_channels * 2, base_channels * 2, base_depth * 1, first=True),
        )

        self.crossStagePartial1 = nn.Sequential(
            DownSampleLayer(base_channels * 2, base_channels * 4),
            WrapLayer(base_channels * 4, base_channels * 4, base_depth * 2),
        )

        self.crossStagePartial2 = nn.Sequential(
            DownSampleLayer(base_channels * 4, base_channels * 8),
            WrapLayer(base_channels * 8, base_channels * 8, base_depth * 8),
        )

        self.crossStagePartial3 = nn.Sequential(
            DownSampleLayer(base_channels * 8, base_channels * 16),
            WrapLayer(base_channels * 16, base_channels * 16, base_depth * 8),
        )

        self.crossStagePartial4 = nn.Sequential(
            DownSampleLayer(base_channels * 16, base_channels * 32),
            WrapLayer(base_channels * 32, base_channels * 32, base_depth * 4),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 32, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.crossStagePartial1(x)
        x = self.crossStagePartial2(x)
        x = self.crossStagePartial3(x)
        x = self.crossStagePartial4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class Focus(nn.Module):
    def __init__(self, c1, c2, k=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = CBM(c1 * 4, c2, k)

    def forward(self, x):
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class CSPDarknetV2(nn.Module):
    def __init__(self, base_channels, base_depth):
        super(CSPDarknetV2, self).__init__()
        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        # -----------------------------------------------#

        CBM.keywords['activation_layer'] = nn.SiLU
        DownSampleLayer = partial(CBM, kernel_size=3, stride=2)

        # -----------------------------------------------#
        #   利用focus网络结构进行特征提取
        #   640, 640, 3 -> 320, 320, 12 -> 320, 320, 64
        # -----------------------------------------------#
        self.stem = Focus(3, base_channels, k=3)
        # -----------------------------------------------#
        #   完成卷积之后，320, 320, 64 -> 160, 160, 128
        #   完成CSPlayer之后，160, 160, 128 -> 160, 160, 128
        # -----------------------------------------------#
        self.crossStagePartial1 = nn.Sequential(
            DownSampleLayer(base_channels, base_channels * 2),
            WrapLayer(base_channels * 2, base_channels * 2, base_depth),
        )
        # -----------------------------------------------#
        #   完成卷积之后，160, 160, 128 -> 80, 80, 256
        #   完成CSPlayer之后，80, 80, 256 -> 80, 80, 256
        # -----------------------------------------------#
        self.crossStagePartial2 = nn.Sequential(
            DownSampleLayer(base_channels * 2, base_channels * 4),
            WrapLayer(base_channels * 4, base_channels * 4, base_depth * 3),
        )

        # -----------------------------------------------#
        #   完成卷积之后，80, 80, 256 -> 40, 40, 512
        #   完成CSPlayer之后，40, 40, 512 -> 40, 40, 512
        # -----------------------------------------------#
        self.crossStagePartial3 = nn.Sequential(
            DownSampleLayer(base_channels * 4, base_channels * 8),
            WrapLayer(base_channels * 8, base_channels * 8, base_depth * 3),
        )
        # -----------------------------------------------#
        #   完成卷积之后，40, 40, 512 -> 20, 20, 1024
        #   完成SPP之后，20, 20, 1024 -> 20, 20, 1024
        #   完成CSPlayer之后，20, 20, 1024 -> 20, 20, 1024
        # -----------------------------------------------#
        self.crossStagePartial4 = nn.Sequential(
            DownSampleLayer(base_channels * 8, base_channels * 16),
            CBM(base_channels * 16, base_channels * 8, 1),
            SPP([5, 9, 3]),
            CBM(base_channels * 8 * 4, base_channels * 16, 1),
            WrapLayer(base_channels * 16, base_channels * 16, base_depth, shortcut=False),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.crossStagePartial1(x)
        x = self.crossStagePartial2(x)
        x = self.crossStagePartial3(x)
        x = self.crossStagePartial4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
