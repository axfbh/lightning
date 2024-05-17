import torch.nn as nn
from torchvision.ops.misc import Conv2dNormActivation
import torch


class InvertedResidual(nn.Module):
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(
            self,
            in_ch,
            exp_ch):
        super().__init__()

        layers = []

        layers.append(
            Conv2dNormActivation(
                in_ch,
                exp_ch,
                kernel_size=1,
                stride=1,
                norm_layer=nn.BatchNorm2d,
                activation_layer=nn.ReLU,
                bias=False,
            )
        )

        # depthwise
        layers.append(
            Conv2dNormActivation(
                exp_ch,
                exp_ch,
                kernel_size=1,
                stride=1,
                groups=exp_ch,
                norm_layer=nn.BatchNorm2d,
                activation_layer=None,
                bias=False,
            )
        )

        self.block = nn.Sequential(*layers)

    def forward(self, input):
        result = self.block(input)

        return result


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplane,
                 plane,
                 group,
                 stride=1,
                 base_width=64,
                 norm_layer=None,
                 activation_layer=None,
                 downsample=None):
        """
        Bottleneck 沙漏模型 (大变小 小变大), 用 于减少参数量

        :param inplane: 输入特征图大小
        :param plane: 最后一个 conv1x1 输出
        :param stride: 步幅
        :param downsample: stride = 2 downsample 会将 残差原图下采样到特征图一样大小，否则 None

        """
        super(Bottleneck, self).__init__()

        self.downsample = downsample

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if activation_layer is None:
            activation_layer = nn.ReLU

        width = int(plane * (base_width / 64.0)) * group

        self.cna1 = Conv2dNormActivation(in_channels=inplane,
                                         out_channels=width,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer,
                                         bias=False,
                                         inplace=True)

        self.cna2 = Conv2dNormActivation(in_channels=width,
                                         out_channels=width,
                                         kernel_size=3,
                                         stride=stride,
                                         padding=1,
                                         groups=group,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer,
                                         bias=False,
                                         inplace=True)

        # self.recover = InvertedResidual(inplane,
        #                                 plane * self.expansion)

        self.recover = Conv2dNormActivation(
            inplane,
            plane * self.expansion,
            kernel_size=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=None,
            inplace=True,
            bias=False,
        )

        self.maxpool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(width, plane * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(plane * self.expansion)
        self.relu = activation_layer(inplace=True)

    def forward(self, x):
        identity = x

        out = self.cna1(x)

        out = self.cna2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity2 = self.recover(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        if out.shape[-2:] != x.shape[-2:]:
            identity2 = self.maxpool(identity2)

        # identity = self.relu1(identity)
        # identity2 = self.relu2(identity2)

        identity = identity + identity2

        out = out + identity

        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 planes,
                 strides,
                 num_classes,
                 width_per_group=64,
                 group=1,
                 norm_layer=None,
                 activation_layer=None):
        """
        :param block: net 模块 （只支持 bottleneck）
        :param layers: 每个 stage 的 产生 block 个数
        :param planes: 每个 stage 的 输出 channel 个数
        :param strides: 每个 stage 的 stride 步幅
        :param width_per_group: 如果使用分组卷积，每个组几张特征图
        :param group: 分组卷积，分几组
        """
        super(ResNet, self).__init__()

        if norm_layer is None:
            self.norm_layer = nn.BatchNorm2d

        if activation_layer is None:
            self.activation_layer = nn.ReLU

        self.inplace = 64
        self.base_width = width_per_group
        self.group = group

        self.cna1 = Conv2dNormActivation(in_channels=3,
                                         out_channels=self.inplace,
                                         kernel_size=7,
                                         stride=2,
                                         padding=3,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer,
                                         bias=False,
                                         inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, planes[0], layers[0], strides[0])
        self.layer2 = self._make_layer(block, planes[1], layers[1], strides[1])
        self.layer3 = self._make_layer(block, planes[2], layers[2], strides[2])
        self.layer4 = self._make_layer(block, planes[3], layers[3], strides[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, plane, layer, stride):
        """

        :param block: bottleneck
        :param plane: bottleneck 特征图输出大小
        :param layer: 创建 bottleneck 个数
        :param stride: 第一个 bottleneck 步幅
        :return:
        """
        downsample = None

        # 确保 残差连接的原图和采样后的特征图，在 channel，size 相同
        if stride != 1 or self.inplace != plane * block.expansion:
            downsample = Conv2dNormActivation(in_channels=self.inplace,
                                              out_channels=plane * block.expansion,
                                              kernel_size=1,
                                              stride=stride,
                                              padding=0,
                                              norm_layer=self.norm_layer,
                                              activation_layer=None,
                                              bias=False)

        layers = nn.Sequential()
        layers.append(
            block(
                self.inplace, plane, self.group, stride, self.base_width, self.norm_layer, self.activation_layer,
                downsample
            )
        )

        self.inplace = plane * block.expansion

        for _ in range(1, layer):
            layers.append(
                block(
                    self.inplace,
                    plane,
                    self.group,
                    base_width=self.base_width,
                    norm_layer=self.norm_layer,
                    activation_layer=self.activation_layer
                )
            )

        return layers

    def forward(self, x):

        x = self.cna1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet50(num_classes) -> ResNet:
    """ Constructs a ResNet-50 models.
    """

    return ResNet(block=Bottleneck,
                  planes=[64, 128, 256, 512],
                  layers=[3, 4, 6, 3],
                  strides=[1, 2, 2, 2],
                  num_classes=num_classes)
