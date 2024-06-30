import torch.nn as nn

from typing import List
from functools import partial

import torch
import math
from utils.utils import make_grid
from torchvision.ops.misc import Conv2dNormActivation
from utils.anchor_utils import AnchorGenerator
from ops.models.misc.dfl import DFL
from utils.boxes import dist2bbox


class YoloV8Head(nn.Module):
    def __init__(self, in_channels_list: List, num_classes: int):
        super(YoloV8Head, self).__init__()

        CBS = partial(Conv2dNormActivation, activation_layer=nn.SiLU)

        self.nc = num_classes  # number of classes
        self.nl = len(in_channels_list)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = num_classes + self.reg_max * 4
        c2 = max(16, in_channels_list[0] // 4, self.reg_max * 4)  # number of outputs per anchor
        c3 = max(in_channels_list[0], min(self.nc, 100))  # channels
        self.reg_head = nn.ModuleList()
        for in_channels in in_channels_list:
            self.reg_head.append(
                nn.Sequential(
                    CBS(in_channels, c2, 3),
                    CBS(c2, c2, 3),
                    nn.Conv2d(c2, 4 * self.reg_max, 1),
                )
            )
        self.cls_head = nn.ModuleList()
        for in_channels in in_channels_list:
            self.cls_head.append(
                nn.Sequential(
                    CBS(in_channels, c3, 3),
                    CBS(c3, c3, 3),
                    nn.Conv2d(c3, self.nc, 1),
                )
            )

        self.anchor = AnchorGenerator([0, 0, 0], [1, 1, 1])
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        self.anchors = AnchorGenerator([0, 0, 0], [1, 1, 1])

        self.reset_parameters()

    def reset_parameters(self):
        stride = [8, 16, 32]
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        for a, b, s in zip(m.reg_head, m.cls_head, stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def forward(self, x: List, H, W):
        imgsze = torch.tensor([H, W], device=x[0].device)

        anchor_points, stride_tensor = (x.transpose(0, 1) for x in self.make_anchors(imgsze, x))

        for i in range(self.nl):
            x[i] = torch.cat((self.reg_head[i](x[i]), self.cls_head[i](x[i])), 1)
        if self.training:  # Training path
            return x

        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), anchor_points.unsqueeze(0)) * stride_tensor.repeat(2, 1)
        y = torch.cat((dbox, cls.sigmoid()), 1)

        return x if self.training else (y, x)

    def make_anchors(self, image_size, preds, offset=0.5):
        anchors, strides = self.anchors(image_size, preds)
        for i in range(len(anchors)):
            anchors[i] = (anchors[i][..., :2] + anchors[i][..., 2:]) / 2
            anchors[i] = anchors[i] / strides[i] + offset
            strides[i] = strides[i].expand(anchors[i].shape[0], -1)
        anchor_points = torch.cat(anchors)
        strides = torch.cat(strides).flip(-1)
        return anchor_points, strides

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=True, dim=1)


class YoloV7Head(nn.Module):
    def __init__(self, in_channels_list: List, anchors: List, num_classes: int):
        super(YoloV7Head, self).__init__()
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.nc = num_classes
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
                b.data[:, 5:5 + self.nc] += math.log(0.6 / (self.nc - 0.99999))
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

                grid = make_grid(ny, nx, 1, 1, device).view((1, 1, ny, nx, 2)).expand(shape)
                anchor_grid = self.anchors[i].view((1, self.na, 1, 1, 2)).expand(shape)

                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), -1)
                xy = (xy * 3 - 1 + grid) * stride  # xy
                wh = (wh * 2) ** 2 * anchor_grid  # wh
                y = torch.cat((xy, wh, conf), 4)

                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1), x)


class YoloV5Head(nn.Module):
    def __init__(self, in_channels_list: List, anchors: List, num_classes: int):
        super(YoloV5Head, self).__init__()
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.nc = num_classes
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
                b.data[:, 5:5 + self.nc] += math.log(0.6 / (self.nc - 0.99999))
                layer.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x: List, H, W):
        z = []  # inference output
        device = self.anchors.device
        imgsze = torch.tensor([W, H], device=device)
        for i in range(self.nl):
            x[i] = self.head[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,75,20,20) to x(bs,3,20,20,25)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx)
            if not self.training:  # inference
                ps = x[i].permute(0, 1, 3, 4, 2).contiguous()

                shape = 1, self.na, ny, nx, 2  # grid shape

                stride = imgsze / torch.tensor([nx, ny], device=device)

                grid = make_grid(ny, nx, 1, 1, device).view((1, 1, ny, nx, 2)).expand(shape)
                anchor_grid = self.anchors[i].view((1, self.na, 1, 1, 2)).expand(shape)

                xy, wh, conf = ps.sigmoid().split((2, 2, self.nc + 1), -1)
                xy = (xy * 3 - 1 + grid) * stride  # xy
                wh = (wh * 2) ** 2 * anchor_grid  # wh
                y = torch.cat((xy, wh, conf), 4)

                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1), x)


class YoloV4Head(nn.Module):
    def __init__(self, in_channels_list: List, anchors: List, num_classes: int):
        super(YoloV4Head, self).__init__()
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.nc = num_classes
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
                b.data[:, 4] += math.log(8 / (416 / s) ** 2)
                b.data[:, 5:5 + self.nc] += math.log(0.6 / (self.nc - 0.99999))
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

                grid = make_grid(ny, nx, 1, 1, device).view((1, 1, ny, nx, 2)).expand(shape)
                anchor_grid = self.anchors[i].view((1, self.na, 1, 1, 2)).expand(shape)

                xy, wh, conf = x[i].split((2, 2, self.nc + 1), -1)
                xy = (xy.sigmoid() + grid) * stride  # xy
                wh = wh.exp() * anchor_grid  # wh
                y = torch.cat((xy, wh, conf.sigmoid()), 4)

                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1), x)
