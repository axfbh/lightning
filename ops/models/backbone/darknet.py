import torch
from torch import nn
import torch.nn.functional as F


# 定义卷积层：经过这个层仅变换通道数
class ConvolutionalLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None):
        super(ConvolutionalLayer, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )


# 残差块结构
class ResidualLayer(nn.Module):
    def __init__(self, in_channels):
        super(ResidualLayer, self).__init__()
        self.reseblock = nn.Sequential(
            ConvolutionalLayer(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0),
            ConvolutionalLayer(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return x + self.reseblock(x)


# 定义残差块叠加层
class wrapLayer(nn.Module):
    def __init__(self, in_channels, count):
        super(wrapLayer, self).__init__()
        self.make_layers = nn.ModuleList()
        for _ in range(count):
            self.make_layers.append(ResidualLayer(in_channels))

    def forward(self, x):
        for conv in self.make_layers:
            x = conv(x)
        return x


# 卷积集合块 卷积神经网络
class ConvolutionalSetLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalSetLayer, self).__init__(
            # 讲解中卷积神经网络的的5个卷积
            ConvolutionalLayer(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            ConvolutionalLayer(out_channels, in_channels, kernel_size=3, stride=1, padding=1),

            ConvolutionalLayer(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            ConvolutionalLayer(out_channels, in_channels, kernel_size=3, stride=1, padding=1),

            ConvolutionalLayer(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )


# 下采样层：既改变通道数，也改变形状
class DownSampleLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(DownSampleLayer, self).__init__(
            ConvolutionalLayer(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
        )


# 上采样层
class UpSampleLayer(nn.Module):
    def __init__(self):
        super(UpSampleLayer, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')


class DarkNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(DarkNet, self).__init__()

        # ----- DarkNet53 ------------
        self.stem = nn.Sequential(
            ConvolutionalLayer(3, 32, 3, 1, 1),
            DownSampleLayer(32, 64),
            ResidualLayer(64),
        )

        self.feature1 = nn.Sequential(
            DownSampleLayer(64, 128),
            wrapLayer(128, 2),
        )

        self.feature2 = nn.Sequential(
            DownSampleLayer(128, 256),
            wrapLayer(256, 8)
        )

        self.feature3 = nn.Sequential(
            DownSampleLayer(256, 512),
            wrapLayer(512, 8)
        )

        self.feature4 = nn.Sequential(
            DownSampleLayer(512, 1024),
            wrapLayer(1024, 4)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.feature1(x)
        x = self.feature2(x)
        x = self.feature3(x)
        x = self.feature4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
