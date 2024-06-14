import torch
import torch.nn as nn
from abc import abstractmethod
from utils.iou import bbox_iou, iou_loss, box_convert
from ops.loss.basic_loss import BasicLoss
from ops.metric.DetectionMetric import smooth_BCE
from utils.anchor_utils import AnchorGenerator, dist2bbox

torch.set_printoptions(precision=4, sci_mode=False)


class YoloAnchorBasedLoss(BasicLoss):
    def __init__(self, model):
        super(YoloAnchorBasedLoss, self).__init__(model)

        m = model.head
        self.cp, self.cn = smooth_BCE(eps=self.hyp.get('label_smoothing', 0.0))

        # yolo 小grid大anchor，大grid小anchor
        self.anchors = m.anchors
        self.nl = m.nl
        self.na = m.na
        self.nc = m.nc

        self.balance = [4.0, 1.0, 0.4]

        # Define criteria
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hyp['cls_pw']], device=self.device))
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hyp['obj_pw']], device=self.device))

    @abstractmethod
    def build_targets(self, targets, grids, image_size):
        raise NotImplemented


class YoloAnchorFreeLoss(BasicLoss):
    def __init__(self, model):
        super(YoloAnchorFreeLoss, self).__init__(model)

        m = model.head
        self.cp, self.cn = smooth_BCE(eps=self.hyp.get('label_smoothing', 0.0))

        self.nl = m.nl
        self.nc = m.nc
        self.no = m.no
        self.reg_max = m.reg_max

        self.anchors = AnchorGenerator([0, 0, 0], [1, 1, 1])

        self.use_dfl = m.reg_max > 1

        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=self.device)

        self.balance = [4.0, 1.0, 0.4]

        # Define criteria
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    @abstractmethod
    def build_targets(self, targets, grids, image_size):
        raise NotImplemented


class YoloLossV3(YoloAnchorBasedLoss):
    def build_targets(self, p, targets, image_size):

        tcls, txy, twh, indices = [], [], [], []
        gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain

        for i in range(self.nl):
            # ----------- grid 大小 -----------
            (nb, _), ng, _ = torch.as_tensor(p[i].shape, device=self.device).split(2)

            # ----------- 图片 与 grid 的比值 -----------
            stride = image_size / ng

            # ----------- 锚框映射到 grid 大小 -----------
            anchor = self.anchors[i] / stride[[1, 0]]

            # ----------- 归一化的 坐标和长宽 -----------
            gain[2:] = (1 / stride)[[1, 0, 1, 0]]

            t = torch.Tensor(size=(0, 7)).to(self.device)

            for si in range(nb):
                tb = targets[targets[:, 0] == si] * gain

                # ----------- 计算 锚框 与 长宽 的 iou -----------
                gwh = tb[:, 4:6]
                iou = bbox_iou(anchor, gwh, in_fmt='wh')
                iou, a = iou.max(0)

                # ------------ 删除小于阈值的框 -------------
                j = iou.view(-1) > self.hyp['anchor_t']
                tb, a = tb[j], a[j]

                tb = torch.cat([tb, a[:, None]], -1)

                t = torch.cat([t, tb], 0)

            # ----------- 分别提取信息，生成 -----------
            b, c = t[:, :2].long().t()

            gxy = t[:, 2:4]

            gwh = t[:, 4:6]

            a = t[:, 6].long()

            gi, gj = gxy.long().t()

            indices.append([b, a, gj, gi])

            txy.append(gxy % 1)

            twh.append(torch.log(gwh / anchor[a]))

            tcls.append(c)

        return tcls, txy, twh, indices

    def forward(self, preds, targets, image_size):
        bs = preds[0].shape[0]

        MSE = nn.MSELoss()

        lcls = torch.zeros(1, dtype=torch.float32, device=self.device)
        lxy = torch.zeros(1, dtype=torch.float32, device=self.device)
        lwh = torch.zeros(1, dtype=torch.float32, device=self.device)
        lobj = torch.zeros(1, dtype=torch.float32, device=self.device)

        tcls, txy, twh, indices = self.build_targets(preds, targets, image_size)

        for i, pi in enumerate(preds):
            b, a, gj, gi = indices[i]

            tobj = torch.zeros_like(pi[..., 0])

            nb = len(b)

            if nb:
                ps = pi[b, a, gj, gi]
                tobj[b, a, gj, gi] = 1

                ptxy = torch.sigmoid(
                    ps[:, 0:2]
                )

                ptwh = ps[:, 2:4]

                # ------------ 计算 偏移量 差值 ------------
                lxy += MSE(ptxy, txy[i])
                lwh += MSE(ptwh, twh[i])

                # ------------ 计算 分类 loss ------------
                if self.nc > 1:
                    t = torch.zeros_like(ps[:, 5:])  # targets
                    t[range(nb), tcls[i] - 1] = 1
                    lcls += self.BCEcls(ps[:, 5:], t)

            # ------------ 计算 置信度 loss ------------
            lobj += self.BCEobj(pi[..., 4], tobj)

        lxy *= self.hyp["lxy"]
        lwh *= self.hyp["lwh"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]

        return (lxy + lwh + lobj + lcls) * bs, torch.cat((lxy, lwh, lobj, lcls)).detach()


class YoloLossV4(YoloAnchorBasedLoss):

    def build_targets(self, p, targets, image_size):
        tcls, tbox, indices, anch = [], [], [], []

        gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain

        for i in range(self.nl):
            # ----------- grid 大小 -----------
            (nb, _), ng, _ = torch.as_tensor(p[i].shape, device=self.device).split(2)

            # ----------- 图片 与 grid 的比值 -----------
            stride = image_size / ng

            # ----------- 锚框映射到 grid 大小 -----------
            anchor = self.anchors[i] / stride[[1, 0]]

            # ----------- box 映射到网格 坐标和长宽 -----------
            gain[2:] = (1 / stride)[[1, 0, 1, 0]]

            t = torch.Tensor(size=(0, 7)).to(self.device)

            for si in range(nb):
                tb = targets[targets[:, 0] == si] * gain

                # ----------- 计算 锚框 与 长宽 的 iou -----------
                gwh = tb[:, 4:6]
                iou = bbox_iou(anchor, gwh, in_fmt='wh')
                iou, a = iou.max(0)

                # ------------ 删除小于阈值的框 -------------
                j = iou.view(-1) > self.hyp['anchor_t']
                tb, a = tb[j], a[j]

                tb = torch.cat([tb, a[:, None]], -1)

                t = torch.cat([t, tb], 0)

            # ----------- 分别提取信息，生成 -----------
            b, c = t[:, :2].long().t()

            gxy = t[:, 2:4]

            gwh = t[:, 4:6]

            a = t[:, 6].long()

            gi, gj = gxy.long().t()

            indices.append([b, a, gj, gi])

            txy = gxy % 1

            tbox.append(torch.cat([txy, gwh], 1))

            anch.append(anchor[a])

            tcls.append(c)

        return tcls, tbox, indices, anch

    def forward(self, preds, targets, image_size):
        bs = preds[0].shape[0]

        lcls = torch.zeros(1, dtype=torch.float32, device=self.device)
        lbox = torch.zeros(1, dtype=torch.float32, device=self.device)
        lobj = torch.zeros(1, dtype=torch.float32, device=self.device)

        tcls, tbox, indices, anchors = self.build_targets(preds, targets, image_size)

        for i, pi in enumerate(preds):
            b, a, gj, gi = indices[i]

            tobj = torch.zeros_like(pi[..., 0])

            nb = len(b)

            if nb:
                ps = pi[b, a, gj, gi]
                tobj[b, a, gj, gi] = 1

                pxy = torch.sigmoid(ps[:, 0:2])

                pwh = torch.exp(ps[:, 2:4]).clamp(max=1E3) * anchors[i]

                pbox = torch.cat([pxy, pwh], 1)

                giou = iou_loss(pbox, tbox[i], in_fmt='cxcywh', GIoU=True)

                lbox += (1.0 - giou).mean()

                if self.nc > 1:
                    t = torch.full_like(ps[:, 5:], self.cn)  # targets
                    t[range(nb), tcls[i] - 1] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)

            lobj += self.BCEobj(pi[..., 4], tobj)

        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()


class YoloLossV5(YoloAnchorBasedLoss):
    def build_targets(self, p, targets, image_size):
        tcls, tbox, indices, anch = [], [], [], []

        gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain

        for i in range(self.nl):
            # ----------- grid 大小 -----------
            (bs, _), ng, _ = torch.as_tensor(p[i].shape, device=self.device).split(2)

            # ----------- 网格 ——----------
            x, y = torch.tensor([[0, 0],
                                 [1, 0],
                                 [0, 1],
                                 [-1, 0],
                                 [0, -1]], device=self.device, dtype=torch.float32).mul(0.5).chunk(2, 1)

            identity = torch.zeros_like(x)

            # ----------- 图片与 grid 的比值 -----------
            stride = image_size / ng

            # ----------- 锚框映射到 grid 大小 -----------
            anchor = self.anchors[i] / stride[[1, 0]]

            na = len(anchor)

            # ----------- 归一化的 坐标和长宽 -----------
            gain[2:] = (1 / stride)[[1, 0, 1, 0]]

            t = torch.Tensor(size=(0, 9)).to(self.device)

            for si in range(bs):
                tb = targets[targets[:, 0] == si] * gain

                nb, cls, cx, cy, gw, gh = tb.unbind(1)

                # ----------- 选择目标点 1 格距离内的网格用于辅助预测 -----------
                tb = torch.stack([nb - identity,
                                  cls - identity,
                                  cx - identity,
                                  cy - identity,
                                  cx - x,
                                  cy - y,
                                  gw - identity,
                                  gh - identity],
                                 -1)

                # j：左格左上角
                j = tb[0, :, 4] % 1 < 0.5
                # k：上格左上角
                k = tb[0, :, 5] % 1 < 0.5
                # l：右格左上角
                l = ~j
                # m：下格左上角
                m = ~k
                j = torch.stack([torch.ones_like(j), j, k, l, m])
                tb = tb[j]
                j = torch.bitwise_and(0 <= tb[..., 4:6], tb[..., 4:6] < ng[[1, 0]]).all(-1)
                tb = tb[j]

                ai = torch.arange(na, device=self.device).view(na, 1).repeat(1, len(tb))

                tb = torch.cat((tb.repeat(na, 1, 1), ai[:, :, None]), 2)

                #  ------------ 选择最大的长宽比，删除小于阈值的框 -------------
                r = tb[..., 6:8] / anchor[:, None]
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']
                tb = tb[j]

                t = torch.cat([t, tb], 0)

            # ----------- 分别提取信息，生成 -----------
            b, c = t[:, :2].long().t()

            gxy = t[:, 2:4]

            gwh = t[:, 6:8]

            gij = t[:, 4:6].long()

            gi, gj = gij.t()

            a = t[:, 8].long()

            indices.append([b, a, gj, gi])

            tbox.append(torch.cat([gxy - gij, gwh], 1))

            anch.append(anchor[a])

            tcls.append(c)

        return tcls, tbox, indices, anch

    def forward(self, preds, targets, image_size):
        bs = preds[0].shape[0]

        lcls = torch.zeros(1, dtype=torch.float32, device=self.device)
        lbox = torch.zeros(1, dtype=torch.float32, device=self.device)
        lobj = torch.zeros(1, dtype=torch.float32, device=self.device)

        tcls, tbox, indices, anchors = self.build_targets(preds, targets, image_size)

        for i, pi in enumerate(preds):

            b, a, gj, gi = indices[i]

            tobj = torch.zeros_like(pi[..., 0])

            nb = len(b)

            if nb:
                ps = pi[b, a, gj, gi]

                pxy = torch.sigmoid(ps[:, 0:2]) * 2 - 0.5

                pwh = (torch.sigmoid(ps[:, 2:4]) * 2) ** 2 * anchors[i]

                pbox = torch.cat([pxy, pwh], 1)

                iou = iou_loss(pbox, tbox[i], in_fmt='cxcywh', CIoU=True)

                lbox += (1 - iou).mean()

                iou = iou.detach().clamp(0).type(tobj.dtype)
                tobj[b, a, gj, gi] = iou  # iou ratio

                if self.nc > 1:
                    t = torch.full_like(ps[:, 5:], self.cn)  # targets
                    t[range(nb), tcls[i] - 1] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss

        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()


class YoloLossV7(YoloAnchorBasedLoss):

    def build_targets(self, p, targets, image_size):
        tcls, tbox, indices, anch = [], [], [], []

        gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain

        for i in range(self.nl):
            # ----------- grid 大小 -----------
            (bs, _), ng, _ = torch.as_tensor(p[i].shape, device=self.device).split(2)

            # ----------- 网格 ——----------
            x, y = torch.tensor([[0, 0],
                                 [1, 0],
                                 [0, 1],
                                 [-1, 0],
                                 [0, -1]], device=self.device, dtype=torch.float32).mul(1.0).chunk(2, 1)

            identity = torch.zeros_like(x)

            # ----------- 图片与 grid 的比值 -----------
            stride = image_size / ng

            # ----------- 锚框映射到 grid 大小 -----------
            anchor = self.anchors[i] / stride[[1, 0]]

            na = len(anchor)

            # ----------- 归一化的 坐标和长宽 -----------
            gain[2:] = (1 / stride)[[1, 0, 1, 0]]

            t = torch.Tensor(size=(0, 9)).to(self.device)

            for si in range(bs):
                tb = targets[targets[:, 0] == si] * gain

                nb, cls, cx, cy, gw, gh = tb.unbind(1)

                # ----------- 选择目标点 1 格距离内的网格用于辅助预测 -----------
                tb = torch.stack([nb - identity,
                                  cls - identity,
                                  cx - identity,
                                  cy - identity,
                                  cx - x,
                                  cy - y,
                                  gw - identity,
                                  gh - identity],
                                 -1)

                j = torch.bitwise_and(0 <= tb[..., 4:6], tb[..., 4:6] < ng[[1, 0]]).all(-1)
                tb = tb[j]

                ai = torch.arange(na, device=self.device).view(na, 1).repeat(1, len(tb))

                tb = torch.cat((tb.repeat(na, 1, 1), ai[:, :, None]), 2)

                #  ------------ 选择最大的长宽比，删除小于阈值的框 -------------
                r = tb[..., 6:8] / anchor[:, None]
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']
                tb = tb[j]

                t = torch.cat([t, tb], 0)

            # ----------- 分别提取信息，生成 -----------
            b, c = t[:, :2].long().t()

            gxy = t[:, 2:4]

            gwh = t[:, 6:8]

            gij = t[:, 4:6].long()

            gi, gj = gij.t()

            a = t[:, 8].long()

            indices.append([b, a, gj, gi])

            tbox.append(torch.cat([gxy - gij, gwh], 1))

            anch.append(anchor[a])

            tcls.append(c)

        return tcls, tbox, indices, anch

    def forward(self, preds, targets, image_size):
        bs = preds[0].shape[0]

        lcls = torch.zeros(1, dtype=torch.float32, device=self.device)
        lbox = torch.zeros(1, dtype=torch.float32, device=self.device)
        lobj = torch.zeros(1, dtype=torch.float32, device=self.device)

        tcls, tbox, indices, anchors = self.build_targets(preds, targets, image_size)

        for i, pi in enumerate(preds):

            b, a, gj, gi = indices[i]

            tobj = torch.zeros_like(pi[..., 0])

            nb = len(b)

            if nb:
                ps = pi[b, a, gj, gi]

                pxy = torch.sigmoid(ps[:, 0:2]) * 3 - 1

                pwh = (torch.sigmoid(ps[:, 2:4]) * 2) ** 2 * anchors[i]

                pbox = torch.cat([pxy, pwh], 1)

                iou = iou_loss(pbox, tbox[i], in_fmt='cxcywh', CIoU=True)

                lbox += (1 - iou).mean()

                iou = iou.detach().clamp(0).type(tobj.dtype)
                tobj[b, a, gj, gi] = iou  # iou ratio

                if self.nc > 1:
                    t = torch.full_like(ps[:, 5:], self.cn)  # targets
                    t[range(nb), tcls[i] - 1] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss

        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()


class YoloLossV8(YoloAnchorFreeLoss):

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)[0]

    def build_targets(self, p, targets, image_size):
        tcls, tbox, indices, anch = [], [], [], []

        bs = p[0].shape[0]

        xyxy = box_convert(targets[:, 2:], in_fmt='cxcywh', out_fmt='xyxy')

        targets = torch.cat([targets[:, :2], xyxy], -1)

        anchors, strides = self.anchors(image_size, p)

        for i in range(self.nl):

            anchor = anchors[i]

            anchor_centers = (anchor[:, :2] + anchor[:, 2:]) / 2 + 0.5 * strides[i]  # N

            pred_distri, pred_scores = p[i].view(bs, self.no, -1).detach().split((self.reg_max * 4, self.nc), 1)

            pred_scores = pred_scores.permute(0, 2, 1).contiguous()

            pred_distri = pred_distri.permute(0, 2, 1).contiguous()

            x, y = anchor_centers.chunk(2, 1)

            identity = torch.zeros_like(x)

            for si in range(bs):

                if si != 2:
                    continue

                tb = targets[targets[:, 0] == si]

                n_boxes = tb.shape[0]

                nb, cls, x0, y0, x1, y1 = tb.unbind(1)

                tb = torch.stack([nb - identity,
                                  cls - identity,
                                  x0 - identity,
                                  y0 - identity,
                                  x - x0,
                                  y - y0,
                                  x1 - x,
                                  y1 - y], dim=-1)

                ltrb_off = tb[..., -4:]
                j = ltrb_off.amin(-1).gt(1e-9)

                pd_scores = pred_scores[si, :, cls.long()].sigmoid()

                pd_bboxes = self.bbox_decode(
                    anchor_centers,
                    pred_distri[si].unsqueeze(0)[..., :self.reg_max * 4]
                ).unsqueeze(1).expand(-1, n_boxes, -1)

                gt_bboxes = torch.stack([x0 - identity,
                                         y0 - identity,
                                         x1 - identity,
                                         y1 - identity], dim=-1)

                iou = iou_loss(gt_bboxes, pd_bboxes, CIoU=True).clamp_(0)

                align_metric = iou.pow(6.0) * pd_scores.pow(1.0)

                topk_metrics, topk_idxs = torch.topk(align_metric, 13, dim=0, largest=True)

                align_metric_max_ind = align_metric.argmax(-1, keepdim=True)

                gt_mask = torch.zeros_like(align_metric, dtype=torch.bool).scatter_(-1, align_metric_max_ind, 1)

                tb = tb[gt_mask]

                tb = tb[topk_idxs.unique()]

                # a = 1
                # topk_metrics, topk_idxs = torch.topk(align_metric, 13, dim=0)

                # tb = tb[align_metric_mask_ind]

        # for i in range(self.nl):
        #     stride = strides[i].flip(0)  # H,W -> W,H
        #
        #     anchor = anchors[i]
        #
        #     anchor_centers = (anchor[:, :2] + anchor[:, 2:]) / 2  # N
        #
        #     # ----------- grid 大小 -----------
        #     (bs, _), ng, _ = torch.as_tensor(p[i].shape, device=self.device).split(2)
        #
        # return tcls, tbox, indices, anch

    def forward(self, preds, targets, image_size):
        bs = preds[0].shape[0]

        lcls = torch.zeros(1, dtype=torch.float32, device=self.device)
        lbox = torch.zeros(1, dtype=torch.float32, device=self.device)
        lobj = torch.zeros(1, dtype=torch.float32, device=self.device)

        tcls, tbox, indices, anchors = self.build_targets(preds, targets, image_size)
