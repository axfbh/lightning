import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial
from torchvision.ops.misc import Conv2dNormActivation
from torchvision.models._utils import IntermediateLayerGetter
from typing import Optional, List, Union
import math

BN = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
CBM = partial(Conv2dNormActivation, bias=False, inplace=True, norm_layer=BN, activation_layer=nn.SiLU)


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
    def __init__(self, c1, c2, ksizes=(5, 9, 13), conv_layer=None,
                 activation_layer=nn.ReLU):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()

        Conv = partial(Conv2dNormActivation,
                       bias=False,
                       inplace=False,
                       norm_layer=nn.BatchNorm2d,
                       activation_layer=activation_layer) if conv_layer is None else conv_layer

        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = SPP(ksizes)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class ResidualLayer(nn.Module):
    def __init__(self, in_ch, out_ch, shortcut=True):
        super(ResidualLayer, self).__init__()
        self.shortcut = shortcut
        self.conv = nn.Sequential(
            CBM(in_ch, out_ch, 1),
            CBM(out_ch, in_ch, 3),
        )

    def forward(self, x):
        return x + self.conv(x) if self.shortcut else self.conv(x)


class WrapLayer(nn.Module):
    def __init__(self, c1, c2, count=1, shortcut=True, first=False):
        super(WrapLayer, self).__init__()
        c_ = c1 if first else c1 // 2
        self.trans_0 = CBM(c1, c_, 1)

        self.trans_1 = CBM(c1, c_, 1)

        self.make_layers = nn.ModuleList()
        for _ in range(count):
            self.make_layers.append(ResidualLayer(c_, c_, shortcut))

        self.trans_cat = CBM(c_ * 2, c2, 1)

    def forward(self, x):
        # ----------- 两分支 -----------
        out0 = self.trans_0(x)
        out1 = self.trans_1(x)

        for conv in self.make_layers:
            out0 = conv(out0)

        out = torch.cat([out0, out1], 1)
        out = self.trans_cat(out)
        return out


class CSPDarknetV2(nn.Module):
    def __init__(self, base_channels, base_depth):
        super(CSPDarknetV2, self).__init__()
        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        # -----------------------------------------------#
        DownSampleLayer = partial(CBM, kernel_size=3, stride=2)

        # -----------------------------------------------#
        #   利用focus网络结构进行特征提取
        #   640, 640, 3 -> 320, 320, 12 -> 320, 320, 64
        # -----------------------------------------------#
        self.stem = CBM(3, 16, 6, 2)
        # -----------------------------------------------#
        #   完成卷积之后，320, 320, 64 -> 160, 160, 128
        #   完成CSPlayer之后，160, 160, 128 -> 160, 160, 128
        # -----------------------------------------------#
        self.crossStagePartial1 = nn.Sequential(
            DownSampleLayer(base_channels, base_channels * 2),
            WrapLayer(base_channels * 2, base_channels * 2, base_depth),
        )
        # -----------------------------------------------#
        #   完成卷积之后，160, 160, 128 -> 80, 80, 256
        #   完成CSPlayer之后，80, 80, 256 -> 80, 80, 256
        # -----------------------------------------------#
        self.crossStagePartial2 = nn.Sequential(
            DownSampleLayer(base_channels * 2, base_channels * 4),
            WrapLayer(base_channels * 4, base_channels * 4, base_depth * 2),
        )

        # -----------------------------------------------#
        #   完成卷积之后，80, 80, 256 -> 40, 40, 512
        #   完成CSPlayer之后，40, 40, 512 -> 40, 40, 512
        # -----------------------------------------------#
        self.crossStagePartial3 = nn.Sequential(
            DownSampleLayer(base_channels * 4, base_channels * 8),
            WrapLayer(base_channels * 8, base_channels * 8, base_depth * 3),
        )
        # -----------------------------------------------#
        #   完成卷积之后，40, 40, 512 -> 20, 20, 1024
        #   完成SPP之后，20, 20, 1024 -> 20, 20, 1024
        #   完成CSPlayer之后，20, 20, 1024 -> 20, 20, 1024
        # -----------------------------------------------#
        self.crossStagePartial4 = nn.Sequential(
            DownSampleLayer(base_channels * 8, base_channels * 16),
            WrapLayer(base_channels * 16, base_channels * 16, base_depth),
            SPPF(base_channels * 16, base_channels * 16, [5], conv_layer=CBM),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.crossStagePartial1(x)
        x = self.crossStagePartial2(x)
        x = self.crossStagePartial3(x)
        x = self.crossStagePartial4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def _cspdarknet_extractor(
        backbone: Union[CSPDarknetV2],
        trainable_layers: int,
        returned_layers: Optional[List[int]] = None):
    # select layers that won't be frozen
    if trainable_layers < 0 or trainable_layers > 6:
        raise ValueError(f"Trainable layers should be in the range [0,6], got {trainable_layers}")
    layers_to_train = ["crossStagePartial4",
                       "crossStagePartial3",
                       "crossStagePartial2",
                       "crossStagePartial1",
                       "stem"][:trainable_layers]

    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if returned_layers is None:
        returned_layers = [2, 3, 4]
    if min(returned_layers) <= 0 or max(returned_layers) >= 5:
        raise ValueError(f"Each returned layer should be in the range [1,5]. Got {returned_layers}")
    return_layers = {f"crossStagePartial{k}": str(v) for v, k in enumerate(returned_layers)}

    return IntermediateLayerGetter(backbone, return_layers)


def make_grid(h, w, sh, sw, dtype, device='cpu'):
    shifts_x = torch.arange(0, w, dtype=dtype, device=device) * sw
    shifts_y = torch.arange(0, h, dtype=dtype, device=device) * sh

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    shifts = torch.stack((shift_x, shift_y), dim=1)
    return shifts


class YoloV5Head(nn.Module):
    def __init__(self, in_channels_list: List, anchors: List, num_classes: int):
        super(YoloV5Head, self).__init__()
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.num_classes = num_classes
        self.no = num_classes + 5
        self.head = nn.ModuleList()
        for in_channels in in_channels_list:
            self.head.append(nn.Conv2d(in_channels, self.na * self.no, 1, 1, 0))

        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)

        self.reset_parameters()

    def reset_parameters(self):
        stride = [8, 16, 32]
        for layer, s in zip(self.head, stride):
            if isinstance(layer, nn.Conv2d):
                b = layer.bias.view(self.na, -1)
                b.data[:, 4] += math.log(8 / (640 / s) ** 2)
                b.data[:, 5:5 + self.num_classes] += math.log(0.6 / (self.num_classes - 0.99999))
                layer.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x: List, H, W):
        z = []  # inference output
        device = self.anchors.device
        imgsze = torch.tensor([W, H], device=device)
        for i in range(self.nl):
            x[i] = self.head[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,75,20,20) to x(bs,3,20,20,25)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:  # inference
                shape = 1, self.na, ny, nx, 2  # grid shape

                stride = imgsze / torch.tensor([nx, ny], device=device)

                grid = make_grid(ny, nx, 1, 1, self.anchors.dtype, device).view((1, 1, ny, nx, 2)).expand(shape)
                anchor_grid = self.anchors[i].view((1, self.na, 1, 1, 2)).expand(shape)

                xy, wh, conf = x[i].sigmoid().split((2, 2, self.num_classes + 1), -1)
                xy = (xy * 2 - 0.5 + grid) * stride  # xy
                wh = (wh * 2) ** 2 * anchor_grid  # wh
                y = torch.cat((xy, wh, conf), 4)

                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1), x)


from ops.loss.yolo_loss import YoloLossV5
from ops.models.detection.utils import Yolo


class YoloV5(Yolo):
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

    def on_fit_start(self) -> None:
        self.compute_loss = YoloLossV5(self)
