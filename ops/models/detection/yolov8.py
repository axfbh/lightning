import torch
from torch import Tensor
import torch.nn as nn
import math
from ops.models.backbone.cspdarknet import CSPDarknetV8, CBM, C2f
from ops.models.backbone.utils import _cspdarknet_extractor
from ops.models.head.yolo_head import YoloV8Head
from ops.models.detection.utils import Yolo
from ops.loss.yolo_loss import YoloLossV8


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class YoloV8(Yolo):
    def __init__(self, num_classes, phi):
        super(YoloV8, self).__init__()

        width_multiple = {'n': 0.25, 's': 0.50, 'm': 0.75, 'l': 1.0, 'x': 1.25}[phi]
        depth_multiple = {'n': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.0, 'x': 1.0}[phi]
        deep_mul = {'n': 1.00, 's': 1.00, 'm': 0.75, 'l': 0.50, 'x': 0.50}[phi]

        base_channels = int(width_multiple * 64)  # 64
        base_depth = max(round(depth_multiple * 3), 1)  # 3
        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        # -----------------------------------------------#

        # ---------------------------------------------------#
        #   生成CSPdarknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   80,80,256
        #   40,40,512
        #   20,20,1024
        # ---------------------------------------------------#
        self.backbone = _cspdarknet_extractor(CSPDarknetV8(base_channels, base_depth, deep_mul), 5)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv3_for_upsample1 = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8,
                                       base_channels * 8,
                                       base_depth,
                                       shortcut=False)

        self.conv3_for_upsample2 = C2f(base_channels * 8 + base_channels * 4,
                                       base_channels * 4,
                                       base_depth,
                                       shortcut=False)

        self.down_sample1 = CBM(base_channels * 4, base_channels * 4, 3, 2)
        self.conv3_for_downsample1 = C2f(base_channels * 8 + base_channels * 4,
                                         base_channels * 8,
                                         base_depth,
                                         shortcut=False)

        self.down_sample2 = CBM(base_channels * 8, base_channels * 8, 3, 2)
        self.conv3_for_downsample2 = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8,
                                         base_channels * 16,
                                         base_depth,
                                         shortcut=False)

        self.head = YoloV8Head([base_channels * 4, base_channels * 8, base_channels * 16], num_classes=num_classes)

    def forward(self, x):
        _, _, H, W = x.size()
        x = self.backbone(x)

        feat1, feat2, feat3 = x['0'], x['1'], x['2']

        P5_upsample = self.upsample(feat3)
        P4 = torch.cat([P5_upsample, feat2], 1)
        P4 = self.conv3_for_upsample1(P4)

        P4_upsample = self.upsample(P4)
        P3 = torch.cat([P4_upsample, feat1], 1)
        P3 = self.conv3_for_upsample2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], 1)
        P4 = self.conv3_for_downsample1(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, feat3], 1)
        P5 = self.conv3_for_downsample2(P5)

        # ---------------------------------------------------#
        #   第三个特征层
        #   P3=(batch_size,75,80,80)
        # ---------------------------------------------------#
        # ---------------------------------------------------#
        #   第二个特征层
        #   P4=(batch_size,75,40,40)
        # ---------------------------------------------------#
        # ---------------------------------------------------#
        #   第一个特征层
        #   P5=(batch_size,75,20,20)
        # ---------------------------------------------------#

        return self.head([P3, P4, P5], H, W)

    def on_fit_start(self) -> None:
        self.compute_loss = YoloLossV8(self)
