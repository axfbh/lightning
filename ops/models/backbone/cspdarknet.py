from typing import Tuple, Union
import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial
from torchvision.ops.misc import Conv2dNormActivation
from ops.models.neck.spp import SPPF

BN = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
CBM = partial(Conv2dNormActivation, bias=False, inplace=True, norm_layer=BN, activation_layer=nn.Mish)


class ResidualLayer(nn.Module):
    def __init__(self, c1, c2, k: Union[Tuple[Tuple, Tuple], Tuple], shortcut: bool = True, expand_ratio=0.5):
        super(ResidualLayer, self).__init__()
        c_ = int(c2 * expand_ratio)
        self.shortcut = shortcut
        self.conv = nn.Sequential(
            CBM(c1, c_, k[0]),
            CBM(c_, c2, k[1])
        )

    def forward(self, x):
        return x + self.conv(x) if self.shortcut else self.conv(x)


class C3(nn.Module):
    def __init__(self, c1, c2, count=1, shortcut=True, expand_ratio=0.5):
        super(C3, self).__init__()
        # c_ = c1 if first else c1 // 2
        c_ = int(c2 * expand_ratio)
        self.trans_0 = CBM(c1, c_, 1)

        self.trans_1 = CBM(c1, c_, 1)

        self.make_layers = nn.Sequential()
        for _ in range(count):
            self.make_layers.append(ResidualLayer(c_, c_, ((3, 3), (1, 1)), shortcut, expand_ratio=1))

        self.trans_cat = CBM(c_ * 2, c2, 1)

    def forward(self, x):
        # ----------- 两分支 -----------
        out0 = self.trans_0(x)
        out1 = self.trans_1(x)

        out0 = self.make_layers(out0)

        out = torch.cat([out0, out1], 1)
        out = self.trans_cat(out)
        return out


class C2f(nn.Module):
    def __init__(self, c1, c2, count=1, shortcut=True, expand_ratio=0.5):
        super(C2f, self).__init__()
        self.c = int(c2 * expand_ratio)
        self.cv1 = CBM(c1, 2 * self.c, 1, 1)
        self.cv2 = CBM((2 + count) * self.c, c2, 1)  # optional act=FReLU(c2)

        self.make_layers = nn.ModuleList()
        for _ in range(count):
            self.make_layers.append(ResidualLayer(self.c, self.c, ((3, 3), (3, 3)), shortcut, expand_ratio=1))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.make_layers)
        return self.cv2(torch.cat(y, 1))


class CSPDarknetV4(nn.Module):
    def __init__(self, base_channels=64, base_depth=3, num_classes=1000):
        super(CSPDarknetV4, self).__init__()

        DownSampleLayer = partial(CBM, kernel_size=3, stride=2)

        self.stem = nn.Sequential(
            CBM(3, base_channels, 3),
            DownSampleLayer(base_channels, base_channels * 2),
            C3(base_channels * 2, base_channels * 2, base_depth * 1, expand_ratio=1),
        )

        self.crossStagePartial1 = nn.Sequential(
            DownSampleLayer(base_channels * 2, base_channels * 4),
            C3(base_channels * 4, base_channels * 4, base_depth * 2),
        )

        self.crossStagePartial2 = nn.Sequential(
            DownSampleLayer(base_channels * 4, base_channels * 8),
            C3(base_channels * 8, base_channels * 8, base_depth * 8),
        )

        self.crossStagePartial3 = nn.Sequential(
            DownSampleLayer(base_channels * 8, base_channels * 16),
            C3(base_channels * 16, base_channels * 16, base_depth * 8),
        )

        self.crossStagePartial4 = nn.Sequential(
            DownSampleLayer(base_channels * 16, base_channels * 32),
            C3(base_channels * 32, base_channels * 32, base_depth * 4),
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


class CSPDarknetV5(nn.Module):
    def __init__(self, base_channels=64, base_depth=3, num_classes=1000):
        super(CSPDarknetV5, self).__init__()

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
        self.stem = CBM(3, base_channels, 6, 2)
        # -----------------------------------------------#
        #   完成卷积之后，320, 320, 64 -> 160, 160, 128
        #   完成CSPlayer之后，160, 160, 128 -> 160, 160, 128
        # -----------------------------------------------#
        self.crossStagePartial1 = nn.Sequential(
            DownSampleLayer(base_channels, base_channels * 2),
            C3(base_channels * 2, base_channels * 2, base_depth),
        )
        # -----------------------------------------------#
        #   完成卷积之后，160, 160, 128 -> 80, 80, 256
        #   完成CSPlayer之后，80, 80, 256 -> 80, 80, 256
        # -----------------------------------------------#
        self.crossStagePartial2 = nn.Sequential(
            DownSampleLayer(base_channels * 2, base_channels * 4),
            C3(base_channels * 4, base_channels * 4, base_depth * 2),
        )

        # -----------------------------------------------#
        #   完成卷积之后，80, 80, 256 -> 40, 40, 512
        #   完成CSPlayer之后，40, 40, 512 -> 40, 40, 512
        # -----------------------------------------------#
        self.crossStagePartial3 = nn.Sequential(
            DownSampleLayer(base_channels * 4, base_channels * 8),
            C3(base_channels * 8, base_channels * 8, base_depth * 3),
        )
        # -----------------------------------------------#
        #   完成卷积之后，40, 40, 512 -> 20, 20, 1024
        #   完成SPP之后，20, 20, 1024 -> 20, 20, 1024
        #   完成CSPlayer之后，20, 20, 1024 -> 20, 20, 1024
        # -----------------------------------------------#
        self.crossStagePartial4 = nn.Sequential(
            DownSampleLayer(base_channels * 8, base_channels * 16),
            C3(base_channels * 16, base_channels * 16, base_depth),
            SPPF(base_channels * 16, base_channels * 16, [5], conv_layer=CBM),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 16, num_classes)

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


class CSPDarknetV8(nn.Module):
    def __init__(self, base_channels: int = 64, base_depth: int = 3, deep_mul=1.0, num_classes=1000):
        super(CSPDarknetV8, self).__init__()

        CBM.keywords['activation_layer'] = nn.SiLU
        DownSampleLayer = partial(CBM, kernel_size=3, stride=2)

        self.stem = CBM(3, base_channels, 3, 2)

        self.crossStagePartial1 = nn.Sequential(
            DownSampleLayer(base_channels, base_channels * 2),
            C2f(base_channels * 2, base_channels * 2, base_depth),
        )

        self.crossStagePartial2 = nn.Sequential(
            DownSampleLayer(base_channels * 2, base_channels * 4),
            C2f(base_channels * 4, base_channels * 4, base_depth),
        )

        self.crossStagePartial3 = nn.Sequential(
            DownSampleLayer(base_channels * 4, base_channels * 8),
            C2f(base_channels * 8, base_channels * 8, base_depth),
        )

        self.crossStagePartial4 = nn.Sequential(
            DownSampleLayer(base_channels * 8, int(base_channels * 16 * deep_mul)),
            C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth),
            SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), [5], conv_layer=CBM),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 16, num_classes)

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


def cpsdarknetv4n(pretrained, norm_layer):
    base_channels = int(0.25 * 32)  # 64
    base_depth = max(round(0.33 * 3), 1)  # 3

    return CSPDarknetV4(base_channels, base_depth)


def cpsdarknetv4s(pretrained, norm_layer):
    base_channels = int(0.25 * 32)  # 64
    base_depth = max(round(0.33 * 3), 1)  # 3

    return CSPDarknetV4(base_channels, base_depth)


def cpsdarknetv4m(pretrained, norm_layer):
    base_channels = int(0.25 * 32)  # 64
    base_depth = max(round(0.33 * 3), 1)  # 3

    return CSPDarknetV4(base_channels, base_depth)


def cpsdarknetv4x(pretrained, norm_layer):
    base_channels = int(0.25 * 32)  # 64
    base_depth = max(round(0.33 * 3), 1)  # 3

    return CSPDarknetV4(base_channels, base_depth)
