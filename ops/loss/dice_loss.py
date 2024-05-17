import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes, multiple=False):
        """
        :param multiple: False 二分类，True 多分类
        """
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.multiple = multiple
        self.smooth = 1e-5

    def forward(self, predict, target):

        if self.multiple:
            predict = F.softmax(predict, dim=1)
            target = F.one_hot(target, self.num_classes)
        else:
            assert predict.size(1) == target.size(), "the size of predict and target must be equal."
            predict = torch.sigmoid(predict)

        num = predict.size(0)

        pre = predict.view(num, -1)
        tar = target.view(num, -1)

        intersection = (pre * tar).sum()  # 利用预测值与标签相乘当作交集
        union = (pre + tar).sum()

        score = 1 - (2 * intersection + self.smooth) / (union + self.smooth)

        return score
