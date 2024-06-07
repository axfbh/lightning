import torch.nn as nn
from torchvision.models.mobilenetv3 import mobilenet_v3_large
from ops.models.backbone.utils import _mobilenet_extractor
from ops.models.misc.atrous import ASPP
from torchvision.ops.misc import Conv2dNormActivation


class LiteSeg(nn.Module):
    def __init__(self, num_classes):
        super(LiteSeg, self).__init__()
        self.backbone = _mobilenet_extractor(mobilenet_v3_large(pretrained=True), 1)

        self.aspp = ASPP(1280, 96, [3, 6, 9])

        self.conv = Conv2dNormActivation(96 * 5 + 1280, 96, 1)

        self.head = nn.Conv2d(96, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.aspp(x)
