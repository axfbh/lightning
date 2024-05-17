import torch.nn as nn
import torch
from abc import abstractmethod
from utils import make_grid
from nms import non_max_suppression
from typing import Sequence
from iou import box_convert
from math import ceil


class PostProcess(nn.Module):
    def __init__(self, device, conf_thres=0.2, iou_thres=0.6):
        super(PostProcess, self).__init__()
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.device = device

    @abstractmethod
    def handle_preds(self, preds: torch.Tensor, anchors: torch.Tensor, image_size: Sequence):
        raise NotImplemented

    def forward(self, preds, anchors, image_size):
        inf_out = self.handle_preds(preds, anchors, image_size)
        return non_max_suppression(inf_out, self.conf_thres, self.iou_thres)


class FcosPostProcess(PostProcess):
    def __init__(self, device, conf_thres=0.2, iou_thres=0.6):
        super(FcosPostProcess, self).__init__(device, conf_thres, iou_thres)

    def handle_preds(self, preds, anchors, image_size):
        inf_out = []

        for i, pred in enumerate(preds):
            # 更改 特征图 形式
            pi = torch.cat(pred, 1)

            pi = pi.permute(0, 2, 3, 1).contiguous()

            bs, ny, nx, nc = pi.size()

            anchor = anchors[i]

            anchor = anchor.reshape([1, ny, nx, 4])

            pxy = 0.5 * (anchor[..., :2] + anchor[..., 2:])

            pwh = anchor[..., 2:] - anchor[..., :2]

            io = pi.clone()

            # 映射到网格
            io[..., 0:2] = pxy - (io[..., 0:2] * pwh)
            io[..., 2:4] = pxy + (io[..., 2:4] * pwh)
            io[..., 0:4] = box_convert(io[..., 0:4], in_fmt='xyxy', out_fmt='cxcywh')

            io[..., 4:] = torch.sigmoid(io[..., 4:])

            cls_scores, _ = torch.max(io[..., 5:], dim=-1)

            io[..., 4] = (io[..., 4] * cls_scores).sqrt()

            io = io.view(bs, -1, nc)

            inf_out.append(io)
        return torch.cat(inf_out, 1)
