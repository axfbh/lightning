import torch
import torch.nn as nn
from utils.loss.loss import dice_loss
from utils.loss.loss import py_sigmoid_focal_loss
from utils.utils import center_of_mass
import mmcv
from utils.utils import arr_permutations, arr_arrange, arr_resize
from functools import partial


class SoloLoss(nn.Module):
    def __init__(self, args,
                 scale_ranges,
                 simga=0.2,
                 gamma=2.0,
                 alpha=0.25):
        super(SoloLoss, self).__init__()
        self.scale_ranges = scale_ranges  # 以面积的方式，决定哪个尺度分割哪个图像
        self.num_classes = args.num_classes
        self.image_size = args.image_size
        self.device = args.device.train
        self.cate_loss = partial(py_sigmoid_focal_loss, weight=None, gamma=gamma, alpha=alpha, reduction='mean')
        self.mask_loss = dice_loss
        self.sigma = simga

    def build_target(self, target, mask, num_grids):
        tmask, indices = [], []

        upsampled_size = mask.shape[-2:]

        for i in range(len(num_grids)):
            ng = num_grids[i]

            anchor = self.scale_ranges[i]

            t, a = target, []

            garea = torch.sqrt(t[:, 4] * t[:, 5])

            j = ((garea >= anchor[0]) & (garea <= anchor[1]))

            t, m = t[j], mask[j]

            # ---------- 使用目标半径缩放后的大小，将覆盖到的网格填充上目标 ------------
            half_ws = 0.5 * t[:, 4] * self.sigma

            half_hs = 0.5 * t[:, 5] * self.sigma

            # ---------- 通过 mask 查找目标的中心点位置 ----------
            center_ws, center_hs = center_of_mass(m)

            # ---------- 中心点归一化 ----------
            coord_ws = ((center_ws / upsampled_size[1]) * ng).int()
            coord_hs = ((center_hs / upsampled_size[0]) * ng).int()

            # ---------- 截断超出网格的值 ----------
            top_box = (((center_hs - half_hs) / upsampled_size[0]) * ng).int().clamp(min=0)
            down_box = (((center_hs + half_hs) / upsampled_size[0]) * ng).int().clamp(max=ng - 1)
            left_box = (((center_ws - half_ws) / upsampled_size[1]) * ng).int().clamp(min=0)
            right_box = (((center_ws + half_ws) / upsampled_size[1]) * ng).int().clamp(max=ng - 1)

            top = torch.maximum(top_box, coord_hs - 1)
            down = torch.minimum(down_box, coord_hs + 1)
            left = torch.maximum(coord_ws - 1, left_box)
            right = torch.minimum(right_box, coord_ws + 1)

            b, c = t[:, :2].long().t()

            # --------- 构建 全排列 ---------
            td = arr_arrange(top.detach().cpu(), down.detach().cpu())
            lr = arr_arrange(left.detach().cpu(), right.detach().cpu())
            rela_coord, fp, m = arr_permutations(td, lr, m, cates=c, val=ng, ids=b)
            indices.append([b, rela_coord, fp])

            tmask.append(m)

        return tmask, indices

    def forward(self,
                mask_preds,
                cate_preds,
                target,
                mask):

        B = cate_preds[0].size(0)

        featmap_sizes = [featmap.size()[-2:] for featmap in mask_preds]

        num_grids = [cate_pred.size(2) for cate_pred in cate_preds]

        lmask = []
        lclsp = []
        lclst = []

        tmask, indices = self.build_target(target, mask, num_grids)

        num_mask = 0

        for i, (pi, pj) in enumerate(zip(cate_preds, mask_preds)):
            b, rela_coord, fp = indices[i]

            num_grid = num_grids[i]

            nb = len(b)

            featmap_size = featmap_sizes[i]

            tobj = torch.zeros_like(pi[:, 0], dtype=torch.int64)

            mask_lable = torch.zeros_like(pj, dtype=torch.uint8)

            mask_ind_label = torch.zeros([B, num_grid ** 2], dtype=torch.bool, device=self.device)

            if nb:
                # ------------- 构建 分类 target ----------
                tobj[fp[:, 0], fp[:, 2], fp[:, 3]] = fp[:, 1].to(self.device)

                # --------- 用于选取，有目标的mask，如果直接通过数组遍历，会得到重叠项 -----------
                mask_ind_label[rela_coord[:, 0], rela_coord[:, 1]] = True

                arr_mask = arr_resize(tmask[i], featmap_size).to(self.device)

                mask_lable[rela_coord[:, 0], rela_coord[:, 1]] = arr_mask

                pmask = pj[mask_ind_label].sigmoid()

                lmask.append(self.mask_loss(pmask, mask_lable[mask_ind_label]))

                num_mask += pmask.size(0)

            lclsp.append(pi.permute(0, 2, 3, 1).reshape(-1, self.num_classes))
            lclst.append(tobj.flatten())

        lclsp = torch.cat(lclsp, 0)
        lclst = torch.cat(lclst, 0)

        # --------- 分类 focal loss ---------
        lcls = self.cate_loss(lclsp.sigmoid(), lclst, lclsp.size(1), avg_factor=num_mask + 1)

        # --------- 分割 dice loss 取均值 ---------
        lmask = torch.cat(lmask, 0).mean() * 3

        loss = lcls + lmask

        return loss, lcls.detach(), lmask.detach()
