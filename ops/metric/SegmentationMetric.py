import torch
from sklearn.metrics import confusion_matrix
import numpy as np
from typing import List
from torchmetrics.functional.segmentation import mean_iou
from ops.utils.torch_utils import from_torch_to_numpy
from torchmetrics.segmentation import MeanIoU


class SegmentationMetric:
    def __init__(self,
                 num_classes,
                 include_background=True):
        self.num_classes = num_classes
        self.include_background = include_background
        self.score = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """

        :param preds: [N, H, W]：sigmoid 后的值`
        :param targets: [N, H, W]： 多分类时，每个通道的类别值需为 1.
        :return:
        """
        m_iou = mean_iou(preds, targets, self.num_classes, self.include_background)
        from_torch_to_numpy(m_iou)

    def compute(self):
        iou = 0
        nt = len(self.stats)
        for m in self.stats:
            iou += m.mean()
        return iou / nt

    def reset(self):
        self.stats.clear()
