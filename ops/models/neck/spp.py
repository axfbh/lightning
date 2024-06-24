import torch.nn as nn
import torch
from torchvision.ops.misc import Conv2dNormActivation
from functools import partial


class SPP(nn.Module):
    def __init__(self, ksizes=(5, 9, 13)):
        """
            SpatialPyramidPooling 空间金字塔池化, SPP 返回包含自己
        """
        super(SPP, self).__init__()
        self.make_layers = nn.ModuleList([nn.MaxPool2d(kernel_size=k, stride=1, padding=(k - 1) // 2) for k in ksizes])

    def forward(self, x):
        return torch.cat([m(x) for m in self.make_layers], 1)


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv3 by Glenn Jocher
    def __init__(self, c1, c2, ksizes=(5, 9, 13), conv_layer=None, activation_layer=nn.ReLU):
        super(SPPF, self).__init__()

        Conv = partial(Conv2dNormActivation,
                       bias=False,
                       inplace=False,
                       norm_layer=nn.BatchNorm2d,
                       activation_layer=activation_layer) if conv_layer is None else conv_layer

        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(ksizes) * 3 + 1), c2, 1, 1)
        self.m = SPP(ksizes)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class SPPCSPC(nn.Module):
    def __init__(self, c1, c2, expand_ratio=0.5, ksizes=(5, 9, 13), conv_layer=None, activation_layer=nn.ReLU):
        super(SPPCSPC, self).__init__()

        Conv = partial(Conv2dNormActivation,
                       bias=False,
                       inplace=False,
                       norm_layer=nn.BatchNorm2d,
                       activation_layer=activation_layer) if conv_layer is None else conv_layer

        c_ = int(2 * c2 * expand_ratio)

        self.cv1 = nn.Sequential(
            Conv(c1, c_, 1),
            Conv(c_, c_, 3),
            Conv(c_, c_, 1),
        )

        self.cv2 = Conv(c1, c_, 1)

        self.spp = SPP(ksizes)

        self.cv3 = nn.Sequential(
            Conv(c_ * 4, c_, 1),
            Conv(c_, c_, 3),
        )

        self.cv4 = Conv(c_ * 2, c2, 1)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)

        x1 = torch.cat([x1, self.spp(x1)], 1)
        x1 = self.cv3(x1)

        x = torch.cat([x1, x2], dim=1)

        return self.cv4(x)
