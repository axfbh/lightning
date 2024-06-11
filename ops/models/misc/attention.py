import torch
import torch.nn as nn
from torch import Tensor

from torchvision.ops.misc import Conv2dNormActivation
from einops import rearrange


class GlobalReceptiveField(nn.Module):
    def __init__(self, c1, c2, stride):
        super(GlobalReceptiveField, self).__init__()
        c_ = int(c1 * 0.5)
        self.stride = stride
        self.conv1 = Conv2dNormActivation(c1, c_, 3, 1)
        self.conv2 = Conv2dNormActivation(c_, c2, 3, 1)

    def forward(self, x: Tensor):
        B, C1, H, W = x.size()
        x = self.conv2(self.conv1(rearrange(x, 'b c (h h1) (w w1) -> b (c h w) h1 w1', h1=self.stride, w1=self.stride)))
        _, C2, *_ = x.size()
        h1 = w1 = C2 // C1 // 2
        return rearrange(x, 'b (c h1 w1) h w -> b c (h h1) (w w1)', h1=h1, w1=w1)
