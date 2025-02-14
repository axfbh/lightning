from typing import Tuple
import torch
from utils.boxes import box_convert


def _loss_inter_union(
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsctk = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk

    return intsctk, unionk


def iou_loss(
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
        in_fmt: str = 'xyxy',
        out_fmt: str = 'xyxy',
        eps: float = 1e-7,
        GIoU=False,
        DIoU=False,
        CIoU=False) -> torch.Tensor:
    boxes1 = box_convert(boxes1, in_fmt, out_fmt)
    boxes2 = box_convert(boxes2, in_fmt, out_fmt)

    intsct, union = _loss_inter_union(boxes1, boxes2)
    iou = intsct / (union + eps)

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    # smallest enclosing box
    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    cw = xc2 - xc1
    ch = yc2 - yc1

    if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
        c_area = cw * ch + 1e-16  # convex area
        return iou - (c_area - union) / (c_area + eps)  # GIoU

    if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
        diagonal_distance_squared = cw ** 2 + ch ** 2 + eps
        x_p = (x2 + x1) / 2
        y_p = (y2 + y1) / 2
        x_g = (x1g + x2g) / 2
        y_g = (y1g + y2g) / 2

        centers_distance_squared = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)

        if DIoU:
            return iou - (centers_distance_squared / diagonal_distance_squared)
        elif CIoU:
            w_gt = x2g - x1g
            h_gt = y2g - y1g
            w_pred = x2 - x1
            h_pred = y2 - y1
            v = (4 / (torch.pi ** 2)) * torch.pow((torch.atan(w_gt / h_gt) - torch.atan(w_pred / h_pred)), 2)
            with torch.no_grad():
                alpha = v / (1 - iou + v + eps)
            return iou - (centers_distance_squared / diagonal_distance_squared + v * alpha)  # CIoU
    else:
        raise TypeError('请选择损失函数')
