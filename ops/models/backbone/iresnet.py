import torch.nn as nn
from torchvision.ops import Conv2dNormActivation
from typing import List, Union
import torch


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride=1,
                 downsample: nn.Module = None,
                 norm_layer: nn.Module = None,
                 start_block=False,
                 end_block=False,
                 exclude_bn0=False):
        """
        ResNet-18 和 ResNet-34 采用这个 block
        :param inplanes: 输入特征
        :param planes: 输出特征
        :param stride:
        :param downsample:
        :param norm_layer:
        :param start_block:
        :param end_block:
        :param exclude_bn0:
        """
        super(BasicBlock, self).__init__()

        self.downsample = downsample
        self.end_block = end_block
        self.start_block = start_block

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # ---------- 只有 second middle block 和 end block 需要 第一个 BN ----------
        self.bn0 = norm_layer(inplanes) if not start_block and not exclude_bn0 else nn.Identity()

        # ---------- 只有 middle block 和 end block 需要 第一个 Relu ----------
        self.relu0 = nn.ReLU(inplace=True) if not start_block else nn.Identity()

        self.cba1 = Conv2dNormActivation(in_channels=inplanes,
                                         out_channels=planes,
                                         kernel_size=3,
                                         stride=stride,
                                         padding=1,
                                         bias=False,
                                         norm_layer=norm_layer,
                                         activation_layer=nn.ReLU)

        self.conv2 = nn.Conv2d(in_channels=planes,
                               out_channels=planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)

        self.bn2 = norm_layer(planes)

        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.bn0(x)
        out = self.relu0(out)

        out = self.cba1(out)

        out = self.conv2(out)

        if self.start_block:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = identity + out

        if self.end_block:
            out = self.bn2(out)
            out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride=1,
                 downsample: nn.Module = None,
                 norm_layer: nn.Module = None,
                 start_block=False,
                 end_block=False,
                 exclude_bn0=False):
        """
        ResNet-50 含 50 以上的采用这个 block
        :param inplanes: 输入特征
        :param planes: 输出特征
        :param stride:
        :param downsample:
        :param norm_layer:
        :param start_block:
        :param end_block:
        :param exclude_bn0:
        """
        super(Bottleneck, self).__init__()

        self.downsample = downsample
        self.end_block = end_block
        self.start_block = start_block

        # ---------- 只有 second middle block 和 end block 需要 第一个 BN ----------
        self.bn0 = norm_layer(inplanes) if not start_block and not exclude_bn0 else nn.Identity()

        # ---------- 只有 middle block 和 end block 需要 第一个 Relu ----------
        self.relu0 = nn.ReLU(inplace=True) if not start_block else nn.Identity()

        self.cba1 = Conv2dNormActivation(in_channels=inplanes,
                                         out_channels=planes,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         bias=False,
                                         norm_layer=norm_layer,
                                         activation_layer=nn.ReLU)

        self.cba2 = Conv2dNormActivation(in_channels=planes,
                                         out_channels=planes,
                                         kernel_size=3,
                                         stride=stride,
                                         padding=1,
                                         bias=False,
                                         norm_layer=norm_layer,
                                         activation_layer=nn.ReLU)

        self.conv3 = nn.Conv2d(in_channels=planes,
                               out_channels=planes * self.expansion,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)

        self.bn3 = norm_layer(planes * self.expansion)

        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.bn0(x)
        out = self.relu0(out)

        out = self.cba1(out)

        out = self.cba2(out)

        out = self.conv3(out)

        if self.start_block:
            out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = identity + out

        if self.end_block:
            out = self.bn3(out)
            out = self.relu3(out)

        return out


class iResNet(nn.Module):
    def __init__(self,
                 block,
                 layers: List,
                 num_classes=1000,
                 norm_layer=None):
        super(iResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        self.inplanes = 64
        self.cba1 = Conv2dNormActivation(in_channels=3,
                                         out_channels=64,
                                         kernel_size=7,
                                         stride=2,
                                         padding=3,
                                         bias=False,
                                         norm_layer=norm_layer,
                                         activation_layer=nn.ReLU)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: Union[BasicBlock], planes: int, blocks: int, stride: int = 1):

        norm_layer = self.norm_layer

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # --------- Improved projection shortcut ---------
            # --------- 只有 Bottleneck 会使用 ---------
            downsample = nn.Sequential(
                # maxpool 3x3 s=2
                nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                # conv 1x1 s=1
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # --------- start ResBlock ---------
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer,
                            start_block=True))

        self.inplanes = planes * block.expansion
        # ---------- first BN elimination ----------
        exclude_bn0 = True
        for _ in range(1, (blocks - 1)):
            # --------- start block 后 bn 状态，建议接下来 非线性 状态 ----------
            # --------- mid block 后 conv 状态，建议接下来 bn 状态 ----------
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer,
                                exclude_bn0=exclude_bn0))
            exclude_bn0 = False

        # --------- end ResBlock ---------
        # --------- mid block 后 conv 状态，建议接下来 bn 状态 ----------
        layers.append(block(self.inplanes, planes, norm_layer=norm_layer, end_block=True, exclude_bn0=exclude_bn0))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.cba1(x)
        x = self.maxpool(x)

        # ---------- cba1 后 x 处于 非线性 状态，建议 conv 下一个状态--------
        x = self.layer1(x)

        # ---------- layer1 后 x 处于 非线性 状态，建议 conv 下一个状态--------
        x = self.layer2(x)

        # ---------- 以此类推 ----------
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


def iresnet18(num_classes=1000, **kwargs):
    """Constructs a iResNet-18 models.
    """
    model = iResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)
    return model


def iresnet50(num_classes=1000, **kwargs):
    """Constructs a iResNet-50 models.
    """
    model = iResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
    return model

