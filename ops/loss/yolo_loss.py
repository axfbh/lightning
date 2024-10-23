from bisect import bisect_left, bisect_right

import torch
import torch.nn as nn
import torch.nn.functional as F

from ops.metric.DetectionMetric import smooth_BCE

from utils.tal import TaskAlignedAssigner, TaskNearestAssigner
from utils.utils import make_grid
from utils.boxes import bbox_iou, iou_loss, box_convert, dist2bbox, bbox2dist

torch.set_printoptions(precision=4, sci_mode=False)


class YoloAnchorBasedLoss(nn.Module):
    def __init__(self, model, topk=3):
        super(YoloAnchorBasedLoss, self).__init__()

        self.hyp = model.hyp
        m = model.head
        self.device = model.device

        self.cp, self.cn = smooth_BCE(eps=self.hyp.get('label_smoothing', 0.0))

        # yolo 小grid大anchor，大grid小anchor
        self.anchors = m.anchors
        self.nl = m.nl
        self.na = m.na
        self.nc = m.nc
        self.no = m.no
        ids = bisect_left([1, 3, 5], topk)
        self.alpha = [0, 2, 3][ids]
        self.gamma = [0, 0.5, 1][ids]

        self.assigner = TaskNearestAssigner(anchor_t=self.hyp['anchor_t'], topk=topk, num_classes=self.nc)

        self.balance = [4.0, 1.0, 0.4]

        # Define criteria
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hyp['cls_pw']], device=model.device))
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hyp['obj_pw']], device=model.device))


class YoloAnchorFreeLoss(nn.Module):
    def __init__(self, model):
        super(YoloAnchorFreeLoss, self).__init__()

        self.hyp = model.hyp
        m = model.head
        self.device = model.device

        self.make_anchors = m.make_anchors
        self.nl = m.nl
        self.nc = m.nc
        self.no = m.no
        self.reg_max = m.reg_max

        self.use_dfl = m.reg_max > 1

        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=self.device)

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(self.device)

        # Define criteria
        self.bce = nn.BCEWithLogitsLoss(reduction="none")


class YoloLossV4To7(YoloAnchorBasedLoss):

    def preprocess(self, targets, batch_size):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            # 获取 target id
            i = targets[:, 0]  # image index
            # 获取每个id的target 个数
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            # 创建矩阵，根据最多的 target 个数
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                # 找打对于 id 的target 索引
                matches = i == j
                # 统计有多少个target
                n = matches.sum()
                # 将 target 填入矩阵中。
                if n:
                    out[j, :n] = targets[matches, 1:] - torch.tensor([1, 0, 0, 0, 0], device=targets.device)
        return out

    def forward(self, preds, batch):
        loss = torch.zeros(3, dtype=torch.float32, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds
        preds = [
            xi.view(feats[0].shape[0], self.na, -1, self.no).split((4, 1, self.nc), -1) for xi in feats
        ]

        batch_size = feats[0].shape[0]
        imgsz = torch.tensor(batch['resized_shape'][0], device=self.device)

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size)
        gt_cls, gt_cxys, gt_whs = targets.split((1, 2, 2), 2)  # cls, xyxy
        mask_gt = gt_cxys.sum(2, keepdim=True).gt_(0)  # [b,n_box,1]

        for i in range(self.nl):
            _, ng, _ = torch.as_tensor(feats[i].shape, device=self.device).split(2)

            stride = (imgsz / ng)[[1, 0]]

            target_box, target_score, anc_wh, fg_mask = self.assigner(
                self.anchors[i] / stride,
                make_grid(*ng, device=self.device),
                gt_cls,
                gt_cxys / stride,
                gt_whs / stride,
                mask_gt
            )

            pred_box, pred_obj, pred_score = preds[i]

            target_obj = torch.zeros_like(pred_obj)

            if fg_mask.any() > 0:
                pxy = pred_box[..., :2].sigmoid() * self.alpha - self.gamma
                pwh = (pred_box[..., 2:].sigmoid() * 2) ** 2 * anc_wh
                pred_box = torch.cat([pxy, pwh], -1)
                iou = iou_loss(pred_box[fg_mask], target_box[fg_mask], in_fmt='cxcywh', CIoU=True)

                loss[0] += (1.0 - iou).mean()

                iou = iou.detach().clamp(0).type(target_obj.dtype)
                target_obj[fg_mask] = iou[:, None]  # iou ratio

                if self.nc > 1:
                    loss[2] += self.BCEcls(pred_score[fg_mask], target_score[fg_mask])

            obji = self.BCEobj(pred_obj, target_obj)
            loss[1] += obji * self.balance[i]  # obj loss

        loss[0] *= self.hyp["box"]
        loss[1] *= self.hyp["obj"]
        loss[2] *= self.hyp["cls"]

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class YoloLossV8(YoloAnchorFreeLoss):

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def preprocess(self, targets, batch_size):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            # 获取 target id
            i = targets[:, 0]  # image index
            # 获取每个id的target 个数
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            # 创建矩阵，根据最多的 target 个数
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                # 找打对于 id 的target 索引
                matches = i == j
                # 统计有多少个target
                n = matches.sum()
                # 将 target 填入矩阵中。
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = box_convert(out[..., 1:5], in_fmt='cxcywh', out_fmt='xyxy')
        return out

    def forward(self, preds, targets, imgsz):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        # make grid
        anchor_points, stride_tensor = self.make_anchors(imgsz, preds)

        targets = self.preprocess(targets, batch_size)
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        # 非填充 bbox 样本索引 mask
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)  # [b,n_box,1]

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor.repeat(1, 2)).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            target_bboxes /= stride_tensor.repeat(1, 2)
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp["box"]  # box gain
        loss[1] *= self.hyp["cls"]  # cls gain
        loss[2] *= self.hyp["dfl"]  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class BboxLoss(nn.Module):
    """
        Criterion class for computing training losses during training.
    """

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask]
        iou = iou_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask], CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
                F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
                + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1)
