import torch
import torch.nn as nn
from ops.utils.torch_utils import de_parallel


class BasicLoss:
    def __init__(self, model):
        super(BasicLoss, self).__init__()
        self.device = model.device
        self.hyp = model.hyp

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplemented
