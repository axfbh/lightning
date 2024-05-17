import torch.nn as nn
from torchvision.ops.misc import Conv2dNormActivation
import torch


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplane,
                 plane,
                 group,
                 stride=1,
                 base_width=64):
        """
        Bottleneck 沙漏模型 (大变小 小变大), 用 于减少参数量

        :param inplane: 输入特征图大小
        :param plane: 最后一个 conv1x1 输出
        :param stride: 步幅
        :param downsample: stride = 2 downsample 会将 残差原图下采样到特征图一样大小，否则 None

        """
        super(Bottleneck, self).__init__()

        width = int(plane * (base_width / 64.0)) * group

        self.cna1 = Conv2dNormActivation(in_channels=inplane,
                                         out_channels=width,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         norm_layer=nn.BatchNorm2d,
                                         activation_layer=nn.ReLU,
                                         bias=False,
                                         inplace=True)

        self.cna2 = Conv2dNormActivation(in_channels=width,
                                         out_channels=width,
                                         kernel_size=3,
                                         stride=stride,
                                         padding=1,
                                         groups=group,
                                         norm_layer=nn.BatchNorm2d,
                                         activation_layer=nn.ReLU,
                                         bias=False,
                                         inplace=True)

        self.cn3 = Conv2dNormActivation(in_channels=width,
                                        out_channels=plane * self.expansion,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        norm_layer=nn.BatchNorm2d,
                                        activation_layer=None,
                                        bias=False,
                                        inplace=False)

    def forward(self, x):
        out = self.cna1(x)

        out = self.cna2(out)

        out = self.cn3(out)

        return out


class DeConv(nn.Module):
    def __init__(self, inplane, plane, group, stride, base_width):
        super(DeConv, self).__init__()
        self.bottleneck = Bottleneck(inplane, plane, group, stride=stride, base_width=base_width)
        self.downsample = Conv2dNormActivation(in_channels=inplane,
                                               out_channels=plane * Bottleneck.expansion,
                                               kernel_size=1,
                                               stride=stride,
                                               padding=0,
                                               norm_layer=nn.BatchNorm2d,
                                               activation_layer=None,
                                               bias=False,
                                               inplace=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x)
        out = self.bottleneck(x)
        out = out + identity
        out = self.relu(out)
        return out


class ResidualLayer(nn.Module):
    def __init__(self, inplane, plane, group, base_width):
        super(ResidualLayer, self).__init__()
        self.bottleneck = Bottleneck(inplane, plane, group, base_width=base_width)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.bottleneck(x)
        out = out + identity
        out = self.relu(out)
        return out


class WrapLayer(nn.Module):
    def __init__(self, inplane, plane, group, layer, base_width):
        super(WrapLayer, self).__init__()
        self.make_layers = nn.ModuleList()
        expansion = Bottleneck.expansion
        inplane = inplane * expansion
        for _ in range(layer):
            self.make_layers.append(ResidualLayer(inplane, plane, group, base_width=base_width))

    def forward(self, x):
        for conv in self.make_layers:
            x = conv(x)
        return x


class ResNet(nn.Module):
    def __init__(self,
                 layers,
                 planes,
                 strides,
                 num_classes,
                 width_per_group=64,
                 group=1):
        super(ResNet, self).__init__()

        self.inplane = 64
        self.base_width = width_per_group
        self.group = group

        self.cna1 = Conv2dNormActivation(in_channels=3,
                                         out_channels=self.inplane,
                                         kernel_size=7,
                                         stride=2,
                                         padding=3,
                                         norm_layer=nn.BatchNorm2d,
                                         activation_layer=nn.ReLU,
                                         bias=False,
                                         inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.models = nn.ModuleList()

        for i in range(len(planes)):
            self.models.append(nn.Sequential(
                # --------- 下采样 的 Bottleneck ---------
                DeConv(self.inplane, planes[i], group, stride=strides[i], base_width=self.base_width),
                # -------- Bottleneck block ----------
                WrapLayer(planes[i], planes[i], group, layer=layers[i] - 1, base_width=self.base_width)
            ))
            self.inplane = planes[i] * Bottleneck.expansion

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.cna1(x)
        x = self.maxpool(x)

        x = self.models[0](x)
        x = self.models[1](x)
        x = self.models[2](x)
        x = self.models[3](x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet50(num_classes) -> ResNet:
    """ Constructs a ResNet-50 models.
    """

    return ResNet(planes=[64, 128, 256, 512],
                  layers=[3, 4, 6, 3],
                  strides=[1, 2, 2, 2],
                  num_classes=num_classes)
