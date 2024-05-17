import torch.nn as nn
import torch


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()


    def activations_hook(self, grad):
        # 获取梯度的钩子
        self.gradients = grad
