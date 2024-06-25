# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
from utils.utils import make_grid
from utils.boxes import iou_loss


class TaskAlignedAssigner(nn.Module):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk=13, num_classes=20, alpha=1.0, beta=6.0, eps=1e-9):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters."""
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """IoU calculation for horizontal bounding boxes."""
        return iou_loss(gt_bboxes, pd_bboxes, CIoU=True).clamp_(0)

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment. Reference code is available at
        https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py.

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)

        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
            target_gt_idx (Tensor): shape(bs, num_total_anchors)
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx).to(device),
                torch.zeros_like(pd_bboxes).to(device),
                torch.zeros_like(pd_scores).to(device),
                torch.zeros_like(pd_scores[..., 0]).to(device),
                torch.zeros_like(pd_scores[..., 0]).to(device),
            )

        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """
            Get in_gts mask, (b, max_num_obj, h*w).
        """
        # 找出满足条件一：正样本 mask
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        # 计算每个正样本的分数
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # 找出满足条件二：正样本 mask
        mask_topk = self.select_topk_candidates(align_metric)
        # mask_topk：满足 topk 的样本 mask
        # mask_in_gts：满足在 gt bbox 内的样本 maks
        # mask_gt：非填充的样本 maks
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        """
        Select the positive anchor center in gt.

        Args:
            xy_centers (Tensor): shape(h*w, 2)
            gt_bboxes (Tensor): shape(b, n_boxes, 4)

        Returns:
            (Tensor): shape(b, n_boxes, h*w)
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        # bbox 的 x1y1 x2y2
        # view(-1,1,4) : 所有 bboxes 汇聚在一个维度
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
        # 计算 anchor 中心点与 gt bboxes 的距离
        # view(bs, n_boxes, n_anchors, -1) ：每个 bboxex 有 n_anchors 个候选样本
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        # 每个 anchor 代表一个样本
        # 保留落在 gt bboxes 内的样本
        return bbox_deltas.amin(3).gt_(eps)  # shape(b, n_boxes, h*w)

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """
        Compute alignment metric given predicted and ground truth bounding boxes.

        Args:
            pd_scores (Tensor): shape(b, h*w, 21)
            pd_bboxes (Tensor): shape(b, h*w, 4)
            gt_labels (Tensor): shape(b, n_boxes, 1)
            gt_bboxes (Tensor): shape(b, n_boxes, 4)
            mask_gt (Tensor): shape(b, n_boxes, h*w)，落在 gt bboxes 内的 mask
        Returns:
            align_metric (Tensor): shape(b, n_boxes, h*w)
            overlaps (Tensor): shape(b, n_boxes, h*w)
        """
        # anchor 个数
        na = pd_bboxes.shape[-2]
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # ind[0] : 每个 anchor 膨胀 n_max_boxes 个 bbox
        # ind[1] : ind[0] 膨胀的同时，每个 bbox 选取自己对应的类别分数
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # 将 pd_boxes 和 gt_boxes 膨胀成每个box的网格维度，并取出对应位置值
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        # 将计算好的 iou 存入对应位置
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, largest=True):
        """
        Select the top-k candidates based on the given metrics.

        Args:
            metrics (Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size,
                              max_num_obj is the maximum number of objects, and h*w represents the
                              total number of anchor points.
            largest (bool): If True, select the largest values; otherwise, select the smallest values.
            mask_gt (Tensor): 非填充的 bbox 样本索引 mask.

        Returns:
            (Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k mask.
        """

        # 每个 bbox 选取网格内 topk 个正样本
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)

        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=metrics.device)
        count_tensor.scatter_(-1, topk_idxs, 1)

        # mask_topk = mask_gt.expand(-1, -1, self.topk).bool()

        # 填充的 bbox 样本置为 0
        # masked_fill_ : 不改变数组的形状，从而将索引置为 0
        # topk_idxs.masked_fill_(~mask_topk, 0)

        # count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=mask_topk.device)
        # ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)

        # # 将每个 box 放入网格位置
        # for k in range(self.topk):
        #     count_tensor.scatter_add_(-1, topk_idxs[:, :, k].unsqueeze(-1), ones)

        # # 只有填充的 bbox 的样本，会在同一个位置多次放入，从而剔除填充 bbox 样本
        # count_tensor.masked_fill_(65 > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the
                                batch size and max_num_obj is the maximum number of objects.
            gt_bboxes (Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (Tensor): Indices of the assigned ground truth objects for positive
                                    anchor points, with shape (b, h*w), where h*w is the total
                                    number of anchor points.
            fg_mask (Tensor): A boolean tensor of shape (b, h*w) indicating the positive
                              (foreground) anchor points.

        Returns:
            (Tuple[Tensor, Tensor, Tensor]): A tuple containing the following tensors:
                - target_labels (Tensor): Shape (b, h*w), containing the target labels for
                                          positive anchor points.
                - target_bboxes (Tensor): Shape (b, h*w, 4), containing the target bounding boxes
                                          for positive anchor points.
                - target_scores (Tensor): Shape (b, h*w, num_classes), containing the target scores
                                          for positive anchor points, where num_classes is the number
                                          of object classes.
        """

        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """
        If an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.

        Args:
            mask_pos (Tensor): shape(b, n_max_boxes, h*w)
            overlaps (Tensor): shape(b, n_max_boxes, h*w)

        Returns:
            target_gt_idx (Tensor): shape(b, h*w)
            fg_mask (Tensor): shape(b, h*w)
            mask_pos (Tensor): shape(b, n_max_boxes, h*w)
        """
        # (b, n_max_boxes, h*w) -> (b, h*w)
        # 将每个网格的bbox合并一起，统计一个网格的bbox数量
        fg_mask = mask_pos.sum(-2)
        # 至少有一个重叠目标
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)
            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)

            # is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            # is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            #
            # mask_pos2 = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)

            is_max_overlaps1 = torch.ones(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps1[mask_multi_gts] = 0
            is_max_overlaps1.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            mask_pos = is_max_overlaps1 * mask_pos
            fg_mask = mask_pos.sum(-2)

        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos


class TaskNearestAssigner(nn.Module):
    def __init__(self, topk=3, num_classes=20, anchor_t=4, num_layers=3, num_acnhors=3):
        super(TaskNearestAssigner, self).__init__()
        self.nl = num_layers
        self.na = num_acnhors
        self.num_classes = num_classes
        self.topk = topk
        self.anchor_t = anchor_t

    def get_targets(self, gt_labels, gt_cxy, gt_wh, grid, target_gt_idx, ng):
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        target_cxys = gt_cxy.view(-1, gt_cxy.shape[-1])[target_gt_idx]
        target_txys = target_cxys - grid
        target_whs = gt_wh.view(-1, gt_wh.shape[-1])[target_gt_idx]

        target_scores = torch.zeros((self.bs, ng, self.num_classes),
                                    dtype=torch.float,
                                    device=target_labels.device)  # (b, h*w, 80)
        target_scores.scatter_(-1, target_labels.unsqueeze(-1), 1)

        return target_scores, target_txys, target_whs

    def get_pos_mask(self, grid, gt_cxy, mask_gt):
        distance_deltas = self.get_box_metrics(grid, gt_cxy)

        distance_metric = distance_deltas.abs().sum(-1)

        mask_topk = self.select_topk_candidates(distance_metric, largest=False)

        mask_pos = mask_topk * mask_gt

        return mask_pos, distance_metric

    def get_box_metrics(self, grid, gt_cxy):
        n_anchors = grid.shape[0]
        gt_cxy = gt_cxy.view(-1, 1, 2)
        distance_deltas = ((grid[None] + 0.5) - gt_cxy).view(self.bs, self.n_max_boxes, n_anchors, -1)
        return distance_deltas

    def select_topk_candidates(self, metrics, largest=True):
        # 每个 bbox 选取网格内 topk 个正样本
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=2, largest=largest)

        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=metrics.device)
        count_tensor.scatter_(-1, topk_idxs, 1)

        return count_tensor.to(metrics.dtype)

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """
        If an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.

        Args:
            mask_pos (Tensor): shape(b, n_max_boxes, h*w)
            overlaps (Tensor): shape(b, n_max_boxes, h*w)

        Returns:
            target_gt_idx (Tensor): shape(b, h*w)
            fg_mask (Tensor): shape(b, h*w)
            mask_pos (Tensor): shape(b, n_max_boxes, h*w)
        """
        # (b, n_max_boxes, h*w) -> (b, h*w)
        # 将每个网格的bbox合并一起，统计一个网格的bbox数量
        fg_mask = mask_pos.sum(-2)
        # 至少有一个重叠目标
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)
            max_overlaps_idx = overlaps.argmax(-2)  # (b, h*w)

            non_overlaps = torch.ones(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            # 重叠位置全部置为 0
            non_overlaps[mask_multi_gts] = 0
            # 重叠位最大分数置置为 1
            non_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            mask_pos = non_overlaps * mask_pos
            fg_mask = mask_pos.sum(-2)

        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos

    @torch.no_grad()
    def forward(self, anc_whs, grids, gt_labels, gt_cxy, gt_wh, strides, mask_gt):
        self.bs = gt_labels.shape[0]
        self.n_max_boxes = gt_labels.shape[1]
        self.device = mask_gt.device

        t_txys, t_whs, t_scores, t_anchs, t_masks = [], [], [], [], []

        for i in range(self.nl):
            grid = grids[i]
            stride = strides[i]
            ng = grid.shape[0]
            anc_wh_nl = anc_whs[i] / stride

            nl_cxys, nl_whs, nl_scores, nl_anchs, nl_masks = [], [], [], [], []

            for j in range(self.na):
                anc_wh = anc_wh_nl[j]
                mask_pos, distance_metric = self.get_pos_mask(grid, gt_cxy / stride, mask_gt)

                target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos,
                                                                                distance_metric,
                                                                                self.n_max_boxes)

                target_scores, target_txys, target_whs = self.get_targets(gt_labels,
                                                                          gt_cxy / stride,
                                                                          gt_wh / stride,
                                                                          grid,
                                                                          target_gt_idx,
                                                                          ng)
                anc_wh = anc_wh.view(1, 1, -1).expand(self.bs, ng, -1)
                r = target_whs / anc_wh
                mask_anc = torch.max(r, 1 / r).max(-1)[0] < self.anchor_t
                fg_mask = fg_mask * mask_anc

                nl_cxys.append(target_txys)
                nl_whs.append(target_whs)
                nl_scores.append(target_scores)
                nl_anchs.append(anc_wh)
                nl_masks.append(fg_mask.bool())

            t_txys.append(torch.stack(nl_cxys, 1))
            t_whs.append(torch.stack(nl_whs, 1))
            t_scores.append(torch.stack(nl_scores, 1))
            t_anchs.append(torch.stack(nl_anchs, 1))
            t_masks.append(torch.stack(nl_masks, 1))

        return t_txys, t_whs, t_scores, t_anchs, t_masks
