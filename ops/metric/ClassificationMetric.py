import numpy as np
import torch
from sklearn import metrics
from typing import List, Union, Any
from math import isnan


class ClassificationMetric:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.accuracy = 0
        self.info_matrix = {'precision': 0,
                            'recall': 0,
                            'f1-score': 0}
        self.num = 0

    def add_batch(self, input, target):
        report = metrics.classification_report(target,
                                               input,
                                               output_dict=True,
                                               zero_division=np.nan)
        self.accuracy += report['accuracy']
        self.num += 1

        weighted_avg = report['weighted avg']
        for key in self.info_matrix.keys():
            self.info_matrix[key] += weighted_avg[key]

    def reset(self):
        self.info_matrix = {'precision': 0,
                            'recall': 0,
                            'f1-score': 0}
        self.num = 0

    def precision_score(self):
        """
        提升精确率是为了不错报,精确率越高越好
        :return:
        """
        return self.info_matrix['precision'] / self.num

    def recall_score(self):
        """
        提升召回率是为了不漏报, 召回率越高，代表实际坏用户被预测出来的概率越高
        :return:
        """
        return self.info_matrix['recall'] / self.num

    def accuracy_score(self):
        return self.accuracy / self.num

    def f1_score(self):
        """
        精确率和召回率的调和平均值, F1 score越高，说明模型越稳健
        :return:
        """
        return self.info_matrix['f1-score'] / self.num
