import os
import math
import platform
import random

import numpy as np

from copy import deepcopy
from pathlib import Path
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.nn.parameter import is_lazy
from torchvision.ops.boxes import box_convert
import torch.distributed as dist

from ops.utils.logging import colorstr
from lightning import Callback
from lightning.fabric.utilities.rank_zero import rank_zero_info
from torch.optim.lr_scheduler import LRScheduler


@torch.no_grad()
def _load_from(model, weight):
    model_state_dict = model.state_dict()

    for k in list(weight.keys()):
        if k in model_state_dict:
            if is_lazy(model_state_dict[k]):
                continue
            shape_model = tuple(model_state_dict[k].shape)
            shape_checkpoint = tuple(weight[k].shape)
            if shape_model != shape_checkpoint:
                weight.pop(k)
        else:
            weight.pop(k)
            print(k)

    model.load_state_dict(weight)


def smart_optimizer(model, name: str = "Adam", lr=0.001, momentum=0.9, decay=1e-5):
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if p_name == "bias":  # bias (no decay)
                g[2].append(p)
            elif p_name == "weight" and isinstance(v, bn):  # weight (no decay)
                g[1].append(p)  # BN weight
            else:
                g[0].append(p)  # Conv weight (with decay)

    if name == "Adam":
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif name == "AdamW":
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == "RMSProp":
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == "SGD":
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f"Optimizer {name} not implemented.")

    optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)

    rank_zero_info(f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
                   f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias')
    return optimizer


def one_cycle(lrf, max_epochs):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / max_epochs)) / 2) * (lrf - 1) + 1


def one_linear(lrf, max_epochs):
    return lambda x: (1 - x / max_epochs) * (1.0 - lrf) + lrf


def smart_scheduler(optimizer, name: str = "Cosine", last_epoch=1, **kwargs):
    if name == "Cosine":
        # T_max: 整个训练过程中的cosine循环次数
        # eta_min: 最小学习率，默认为0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               last_epoch=last_epoch,
                                                               **kwargs)
    elif name == "MultiStep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         last_epoch=last_epoch,
                                                         **kwargs)
    elif name == "Polynomial":
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer,
                                                          last_epoch=last_epoch,
                                                          **kwargs)
    elif name == 'OneLinearLR':
        fn = one_linear(**kwargs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      last_epoch=last_epoch,
                                                      lr_lambda=fn)

    else:
        raise NotImplementedError(f"Optimizer {name} not implemented.")

    args = {k: v for k, v in kwargs.items()}
    rank_zero_info(f"{colorstr('scheduler:')} {type(scheduler).__name__}(" + ", ".join(
        f"{k}={v}" for k, v in args.items()) + ")")
    return scheduler


@contextmanager
def torch_distributed_zero_first(local_rank: int, num_nodes: int):
    # Decorator to make all processes in distributed training wait for each local_master to do something
    if local_rank not in [-1, 0]:
        dist.barrier()
    yield
    if local_rank == 0 and num_nodes > 1:
        dist.barrier()


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe


def is_parallel(model):
    # Returns True if models is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a models: returns single-GPU models if models is of type DP or DDP
    return model.module if is_parallel(model) else model


def output_to_target(output, max_det=300):
    # Convert models output to target format [batch_id, class_id, x, y, w, h, conf] for plotting
    targets = []
    for i, o in enumerate(output):
        box, conf, cls = o[:max_det, :6].cpu().split((4, 1, 1), 1)
        j = torch.full((conf.shape[0], 1), i)
        targets.append(torch.cat((j, cls, box_convert(box, 'xyxy', 'cxcywh'), conf), 1))
    return torch.cat(targets, 0).numpy()


def from_torch_to_numpy(feat: torch.Tensor):
    return feat.cpu().numpy()


class ModelEMA:
    """Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the models state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        super(ModelEMA, self).__init__()
        # Create EMA
        self.ema = deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        self.updates += 1
        d = self.decay(self.updates)

        msd = model.state_dict()  # models state_dict
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:  # true for FP16 and FP32
                v *= d
                v += (1 - d) * msd[k].detach()

    def __call__(self, *args, **kwargs):
        return self.ema(*args, **kwargs)
