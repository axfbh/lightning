import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        """
        Single-label target should be make as [B,H,W]. Multi-label target must be make as [B,H,W,C] or [B,C,H,W]
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        predict = torch.sigmoid(predict)

        num = predict.size(0)

        pre = predict.view(num, -1)
        tar = target.view(num, -1)

        intersection = (pre * tar).sum()  # 利用预测值与标签相乘当作交集
        union = (pre + tar).sum()

        score = 1 - (2 * intersection + self.smooth) / (union + self.smooth)

        return score
