import torch
from sklearn.metrics import confusion_matrix
import numpy as np
from typing import List


class SegmentationMetric:
    def __init__(self, num_classes, thresh=0.5):
        self.num_classes = num_classes
        self.thresh = thresh

    def preprocess_tensor_to_numpy(self, input: torch.Tensor, target: torch.Tensor):
        y_pre = input.detach().ge(self.thresh).cpu().numpy().reshape(-1)
        y_true = target.ge(self.thresh).cpu().numpy().reshape(-1)
        return y_true, y_pre

    def fit(self, input: torch.Tensor, target: torch.Tensor):
        """

        :param input: [N, classes, H, W]：sigmoid 后的值
        :param target: [N, classes, H, W]： 多分类时，每个通道的类别值需为 1.
        :return:
        """
        self.confusions = []

        # -------------- 1 个类别时候的处理 --------------
        if input.size(1) == 1:
            y_true, y_pre = self.preprocess_tensor_to_numpy(input, target)
            self.confusions.append(confusion_matrix(y_true, y_pre))
        else:
            num_classes = input.size(1)
            for i in range(num_classes):
                y_true, y_pre = self.preprocess_tensor_to_numpy(input[:, i], target[:, i])
                self.confusions.append(confusion_matrix(y_true, y_pre))

    def mean_iou(self) -> List:
        miou = []
        for conf in self.confusions:
            TN, FP, FN, TP = conf[0, 0], conf[0, 1], conf[1, 0], conf[1, 1]
            miou.append(float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0)
        return miou if self.num_classes > 2 else miou[0]

    def sensitivity(self) -> List:
        recall = []
        for conf in self.confusions:
            TN, FP, FN, TP = conf[0, 0], conf[0, 1], conf[1, 0], conf[1, 1]
            recall.append(float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0)
        return recall if self.num_classes > 2 else recall[0]

    def pixel_accuracy(self) -> List:
        accuracy = []
        for conf in self.confusions:
            TN, FP, FN, TP = conf[0, 0], conf[0, 1], conf[1, 0], conf[1, 1]
            accuracy.append(float(TN + TP) / float(np.sum(conf)) if float(np.sum(conf)) != 0 else 0)
        return accuracy if self.num_classes > 2 else accuracy[0]

    def specificity(self) -> List:
        neg = []
        for conf in self.confusions:
            TN, FP, FN, TP = conf[0, 0], conf[0, 1], conf[1, 0], conf[1, 1]
            neg.append(float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0)
        return neg if self.num_classes > 2 else neg[0]

    def f1_or_dsc(self) -> List:
        f1 = []
        for conf in self.confusions:
            TN, FP, FN, TP = conf[0, 0], conf[0, 1], conf[1, 0], conf[1, 1]
            f1.append(float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0)
        return f1 if self.num_classes > 2 else f1[0]