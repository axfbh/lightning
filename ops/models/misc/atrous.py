from typing import List
from functools import partial

import torch
import torch.nn as nn
from torchvision.ops.misc import Conv2dNormActivation


class ASPP(nn.Module):
    def __init__(self, c1, c2, rates: List):
        super(ASPP, self).__init__()

        self.conv = Conv2dNormActivation(c1, c2, 1)

        self.groups_conv = nn.Sequential()

        for r in rates:
            self.groups_conv.append(nn.Sequential(Conv2dNormActivation(c1, c2, 3, dilation=r),
                                                  Conv2dNormActivation(c2, c2, 3)))

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             Conv2dNormActivation(c1, c2, 1),
                                             nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True))

    def forward(self, x):
        res = [self.conv(x)]
        for conv in self.groups_conv:
            res.append(conv(x))

        res.append(self.global_avg_pool(x))

        return torch.cat(res, 1)
