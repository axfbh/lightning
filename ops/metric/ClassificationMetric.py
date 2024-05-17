import numpy as np
import torch
from sklearn import metrics
from typing import List, Union, Any
from math import isnan


class ClassificationMetric:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def fit(self, input: torch.Tensor, target: torch.Tensor):
        self.input = torch.softmax(input, 1).argmax(1).cpu().numpy()
        self.target = target.numpy()
        if self.num_classes > 2:
            report = metrics.classification_report(self.target,
                                                   self.input,
                                                   output_dict=True,
                                                   zero_division=np.nan)
            self.accuracy = report['accuracy']
            self.weighted_avg = report['weighted avg']

    def accuracy_score(self):
        if self.num_classes > 2:
            return self.accuracy
        return metrics.accuracy_score(self.target, self.input)

    def precision_score(self):
        """
        提升精确率是为了不错报,精确率越高越好
        :return:
        """
        if self.num_classes > 2:
            return self.weighted_avg['precision']
        return metrics.precision_score(self.target, self.input)

    def recall_score(self):
        """
        提升召回率是为了不漏报, 召回率越高，代表实际坏用户被预测出来的概率越高
        :return:
        """
        if self.num_classes > 2:
            return self.weighted_avg['recall']
        return metrics.recall_score(self.target, self.input)

    def f1_score(self):
        """
        精确率和召回率的调和平均值, F1 score越高，说明模型越稳健
        :return:
        """
        if self.num_classes > 2:
            return self.weighted_avg['f1-score']
        return metrics.f1_score(self.target, self.input)