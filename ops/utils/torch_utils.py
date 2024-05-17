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

from ops.utils.logging import LOGGER, colorstr
from lightning import Callback
from lightning.fabric.utilities.rank_zero import rank_zero_only
from ops.utils.logging import logger_info_rank_zero_only


def select_device(device="", batch_size=0, newline=True):
    # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
    s = f"Python-{platform.python_version()} torch-{torch.__version__} "
    device = str(device).strip().lower().replace("cuda:", "").replace("none", "")  # to string, 'cuda:0' to '0'
    cpu = device == "cpu"
    mps = device == "mps"  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(
            device.replace(",", "")
        ), f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(",") if device else "0"  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f"batch-size {batch_size} not multiple of GPU count {n}"
        space = " " * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = "cuda:0"
    elif mps and getattr(torch, "has_mps", False) and torch.backends.mps.is_available():  # prefer MPS if available
        s += "MPS\n"
        arg = "mps"
    else:  # revert to CPU
        s += "CPU\n"
        arg = "cpu"

    if not newline:
        s = s.rstrip()
    LOGGER.info(s)
    return torch.device(arg)


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

    logger_info_rank_zero_only(f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
                               f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias')
    return optimizer


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
    elif name == "OneCycleLR":
        # max_lr (float or list) – Upper learning rate boundaries in the cycle for each parameter group.
        # anneal_strategy (str) – {‘cos’, ‘linear’} Specifies the annealing strategy: “cos” for cosine annealing,
        # “linear” for linear annealing. Default: ‘cos’
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        last_epoch=last_epoch,
                                                        **kwargs)
    else:
        raise NotImplementedError(f"Optimizer {name} not implemented.")

    args = {k: v for k, v in kwargs.items()}
    logger_info_rank_zero_only(f"{colorstr('scheduler:')} {type(scheduler).__name__}(" + ", ".join(
        f"{k}={v}" for k, v in args.items()) + ")")
    return scheduler


def smart_resume(model, optimizer, ema=None, epochs=300, resume=False, save_path: Path = None):
    last_epoch = -1
    last_iter = -1
    best_fitness = None
    if not resume or save_path is None or not save_path.is_file():
        if resume:
            LOGGER.warning(
                f"{colorstr('Warning:')} Noting to resume"
            )
        start_epoch = last_epoch + 1
        return best_fitness, last_iter, last_epoch, start_epoch, epochs

    save_dict = torch.load(save_path, map_location='cpu')

    # ------------ resume models ------------
    model_param = save_dict['models'].state_dict()
    _load_from(model, model_param)

    last_epoch = save_dict.get('last_epoch', last_epoch)
    last_iter = save_dict.get('last_iter', last_iter)
    start_epoch = last_epoch + 1

    best_fitness = save_dict.get('best_fitness', best_fitness)

    LOGGER.info(
        f"{colorstr('models loaded:')} Resuming training from {save_path} from epoch {start_epoch} to {epochs} total epochs"
    )

    # ------------ resume ema ------------
    ema_param = save_dict.get('ema', None)

    if ema is not None and ema_param is not None:
        ema.ema.load_state_dict(ema_param.state_dict())  # EMA
        ema.updates = save_dict["updates"]

    # ------------ resume optimizer ------------
    optim_param = save_dict.get('optimizer', None)
    optim_name = save_dict.get('optimizer_name', None)

    if optim_name == optimizer.__class__.__name__ and optim_param is not None:
        optimizer.load_state_dict(optim_param)
        for param in optimizer.param_groups:
            LOGGER.info(
                f"{colorstr('optimizer loaded:')} {type(optimizer).__name__}(lr={param['lr']}) with parameter groups"
                f"{len(param)} weight(decay={param['weight_decay']})"
            )
    else:
        LOGGER.warning(
            f"{colorstr('Warning:')} Cannot loaded the optimizer parameter, but it doesnt affect the models working."
        )

    return best_fitness, last_iter, last_epoch, start_epoch, epochs


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
