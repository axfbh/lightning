import torch
import math
from torchvision.ops.boxes import box_convert, box_iou
from typing import Tuple


def _wh_to_coor(box: torch.Tensor):
    new_box_max = box / 2
    new_box_min = -new_box_max

    new_box = torch.hstack((new_box_min, new_box_max))
    return new_box


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """
        Transform distance(ltrb) to box(xywh or xyxy).
    """
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """
        Transform bbox(xyxy) to dist(ltrb).
    """
    x1y1, x2y2 = bbox.chunk(2, -1)
    # 获取候选框（lt,rb）
    # clamp：将 lt,rb 长度限制在 16 内
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # dist (lt, rb)


def bbox_iou(boxes1: torch.Tensor,
             boxes2: torch.Tensor,
             in_fmt: str = 'xyxy',
             out_fmt: str = 'xyxy'):
    """
    Args:
         each boxes1 match all the boxes2, boxes1. the shape of box either as: [4] or [w, h]
         boxes1 (Tensor[N, 4]): first set of boxes
         boxes2 (Tensor[M, 4]): second set of boxes
         in_fmt (str): Input format of given boxes. Supported formats are ['wh','xyxy', 'xywh', 'cxcywh'].
         out_fmt (str): Output format of given boxes. Supported formats are ['xyxy', 'xywh', 'cxcywh']

    Returns:
           Tensor[N, M]: the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    if in_fmt == 'wh':
        boxes1 = _wh_to_coor(boxes1)
        boxes2 = _wh_to_coor(boxes2)
    else:
        boxes1 = box_convert(boxes1, in_fmt, out_fmt)
        boxes2 = box_convert(boxes2, in_fmt, out_fmt)

    return box_iou(boxes1, boxes2)



