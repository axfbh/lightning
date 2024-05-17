import torch
import torch.nn as nn
import math
from ops.models.backbone.cspdarknet import CSPDarknetV2, CBM, WrapLayer
from ops.models.backbone.utils import _cspdarknet_extractor
from ops.models.head.yolo_head import YoloV5Head


class YoloV5(nn.Module):
    def __init__(self, anchors, num_classes, depth_multiple, width_multiple):
        super(YoloV5, self).__init__()

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
        self.backbone = _cspdarknet_extractor(CSPDarknetV2(base_channels, base_depth), 5)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_for_feat3 = CBM(base_channels * 16, base_channels * 8, 1, 1)
        self.conv3_for_upsample1 = WrapLayer(base_channels * 16, base_channels * 8, base_depth, shortcut=False)

        self.conv_for_feat2 = CBM(base_channels * 8, base_channels * 4, 1, 1)
        self.conv3_for_upsample2 = WrapLayer(base_channels * 8, base_channels * 4, base_depth, shortcut=False)

        self.down_sample1 = CBM(base_channels * 4, base_channels * 4, 3, 2)
        self.conv3_for_downsample1 = WrapLayer(base_channels * 8, base_channels * 8, base_depth, shortcut=False)

        self.down_sample2 = CBM(base_channels * 8, base_channels * 8, 3, 2)
        self.conv3_for_downsample2 = WrapLayer(base_channels * 16, base_channels * 16, base_depth, shortcut=False)

        self.head = YoloV5Head([base_channels * 4, base_channels * 8, base_channels * 16],
                               anchors,
                               num_classes)

    def forward(self, x):
        _, _, H, W = x.size()
        x = self.backbone(x)

        feat1, feat2, feat3 = x['0'], x['1'], x['2']

        P5 = self.conv_for_feat3(feat3)
        P5_upsample = self.upsample(P5)
        P4 = torch.cat([P5_upsample, feat2], 1)
        P4 = self.conv3_for_upsample1(P4)

        P4 = self.conv_for_feat2(P4)
        P4_upsample = self.upsample(P4)
        P3 = torch.cat([P4_upsample, feat1], 1)
        P3 = self.conv3_for_upsample2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], 1)
        P4 = self.conv3_for_downsample1(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], 1)
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
